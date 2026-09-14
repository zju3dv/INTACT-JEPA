"""Train the clean, end-to-end INTACT model."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from sdpa_policy import (
    sdpa_kernel_context,
    sdpa_policy_metadata,
    validate_sdpa_backend,
)
from utils import (
    PreviousActionDataset,
    RunMetadataCallback,
    SaveCkptCallback,
    TrainingProgressCallback,
    get_column_normalizer,
    get_column_stats,
    get_img_preprocessor,
)


def predict_adjacent_latents(model, embeddings, action_embeddings, history_size):
    """Predict every adjacent transition with at most ``history_size`` context.

    The short 1/2-frame prefixes are evaluated together. Remaining full-history
    windows are folded into the batch dimension, so an eight-frame N=3 window
    receives all seven Forward targets with only two Predictor calls.
    """
    transitions = embeddings.size(1) - 1
    if history_size < 2 or transitions < 1:
        raise ValueError("Forward training needs N>=2 and at least two frames")
    if embeddings.shape[:2] != action_embeddings.shape[:2]:
        raise ValueError("Embedding and action-embedding windows must match in B,T")

    prefix_length = min(history_size, transitions)
    prefix_predictions = model.predict(
        embeddings[:, :prefix_length], action_embeddings[:, :prefix_length]
    )
    if transitions == prefix_length:
        return prefix_predictions

    tail_indices = range(prefix_length, transitions)
    tail_embeddings = torch.cat(
        [
            embeddings[:, index - history_size + 1 : index + 1]
            for index in tail_indices
        ],
        dim=0,
    )
    tail_actions = torch.cat(
        [
            action_embeddings[:, index - history_size + 1 : index + 1]
            for index in tail_indices
        ],
        dim=0,
    )
    tail_predictions = model.predict(tail_embeddings, tail_actions)[:, -1]
    tail_predictions = tail_predictions.reshape(
        transitions - prefix_length, embeddings.size(0), embeddings.size(-1)
    ).transpose(0, 1)
    return torch.cat([prefix_predictions, tail_predictions], dim=1)


def construct_intents(
    embeddings: torch.Tensor, intent_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct attached local intents and detached deployment intents."""
    current = embeddings[:, :-1]
    successor = embeddings[:, 1:]
    local = successor - current
    goal = embeddings[:, -1:].detach().expand_as(current)
    displacement = goal - current
    if intent_mode == "waypoint":
        remaining = torch.arange(
            current.size(1),
            0,
            -1,
            device=embeddings.device,
            dtype=embeddings.dtype,
        ).view(1, -1, 1)
        deployment = displacement / remaining
    elif intent_mode == "goal_displacement":
        deployment = displacement
    else:
        raise ValueError(f"Unknown intent mode: {intent_mode}")
    return local, deployment


def paired_intent_action_loss(
    model, embeddings, previous_actions, target_actions, cfg
):
    """Compute seven physical and seven paired deployment NLLs.

    Both conditions cover every adjacent transition in the eight-frame window,
    use the same demonstrated target action, and are folded into one Actor call.
    """
    local_weight = float(cfg.loss.intent.local_weight)
    goal_weight = float(cfg.loss.intent.goal_weight)
    total_weight = local_weight + goal_weight
    if total_weight <= 0:
        zero = embeddings.new_zeros(())
        return {
            "intent_loss": zero,
            "local_nll": zero,
            "goal_nll": zero,
            "local_mae": zero,
            "goal_mae": zero,
            "intent_mae": zero,
        }

    local_intent, goal_intent = construct_intents(embeddings, model.intent_mode)
    transition_count = local_intent.size(1)
    if transition_count < 1:
        raise ValueError("Actor training requires at least one transition")
    expected_action_shape = (embeddings.size(0), transition_count)
    if previous_actions.ndim != 3 or previous_actions.shape[:2] != expected_action_shape:
        raise ValueError(
            "Previous actions must have shape [B,T-1,A] for all physical transitions"
        )
    if target_actions.ndim != 3 or target_actions.shape != previous_actions.shape:
        raise ValueError(
            "Target actions must match the real previous-action tensor"
        )
    if not torch.isfinite(previous_actions).all() or not torch.isfinite(
        target_actions
    ).all():
        raise ValueError("Actor training pairs contain non-finite actions")

    current = embeddings[:, :-1]
    family_names = []
    family_rows = []
    family_row_counts = []
    if local_weight > 0:
        family_names.append("local")
        family_rows.append(
            (current, local_intent, previous_actions, target_actions)
        )
        family_row_counts.append(current.numel() // current.size(-1))
    if goal_weight > 0:
        family_names.append("goal")
        family_rows.append((current, goal_intent, previous_actions, target_actions))
        family_row_counts.append(current.numel() // current.size(-1))

    def fold(tensor):
        return tensor.reshape(-1, 1, tensor.size(-1))

    stats = model.action_nll(
        z=torch.cat([fold(row[0]) for row in family_rows], dim=0),
        intent=torch.cat([fold(row[1]) for row in family_rows], dim=0),
        previous_action=torch.cat([fold(row[2]) for row in family_rows], dim=0),
        target_action=torch.cat([fold(row[3]) for row in family_rows], dim=0),
        reduction="none",
    )

    nll_by_family = {
        name: values.mean()
        for name, values in zip(
            family_names,
            stats["nll"].split(family_row_counts, dim=0),
            strict=True,
        )
    }
    mae_by_family = {
        name: values.mean()
        for name, values in zip(
            family_names,
            stats["mae"].split(family_row_counts, dim=0),
            strict=True,
        )
    }
    zero = embeddings.new_zeros(())
    local_nll = nll_by_family.get("local", zero)
    goal_nll = nll_by_family.get("goal", zero)
    local_mae = mae_by_family.get("local", zero)
    goal_mae = mae_by_family.get("goal", zero)
    weighted_nll = local_weight * local_nll + goal_weight * goal_nll
    weighted_mae = local_weight * local_mae + goal_weight * goal_mae

    return {
        "intent_loss": weighted_nll,
        "local_nll": local_nll.detach(),
        "goal_nll": goal_nll.detach(),
        "local_mae": local_mae.detach(),
        "goal_mae": goal_mae.detach(),
        "intent_mae": (weighted_mae / total_weight).detach(),
    }


def intact_forward(self, batch, stage, cfg):
    if "previous_action" not in batch:
        raise KeyError("Boundary-aware previous_action is required")
    if not torch.isfinite(batch["action"]).all() or not torch.isfinite(
        batch["previous_action"]
    ).all():
        raise ValueError("Training action pairs contain non-finite values")
    output = self.model.encode(batch)
    embeddings = output["emb"]
    predictions = predict_adjacent_latents(
        self.model, embeddings, output["act_emb"], cfg.history_size
    )
    targets = embeddings[:, 1:]

    pred_loss = (predictions - targets).square().mean()
    sigreg_loss = self.sigreg(embeddings.transpose(0, 1))
    intent = paired_intent_action_loss(
        self.model,
        embeddings,
        batch["previous_action"][:, :-1],
        batch["action"][:, :-1],
        cfg,
    )
    loss = (
        float(cfg.loss.forward_weight) * pred_loss
        + float(cfg.loss.sigreg.weight) * sigreg_loss
        + intent["intent_loss"]
    )

    metrics = {
        "loss": loss,
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        **intent,
    }
    self.log_dict(
        {f"{stage}/{key}": value for key, value in metrics.items()},
        on_step=stage == "train",
        on_epoch=True,
        prog_bar=key_in_progress_bar(metrics),
        sync_dist=True,
    )
    return metrics


def key_in_progress_bar(metrics):
    # Lightning applies this flag to the whole mapping; keep the mapping compact.
    return "loss" in metrics


def validate_training_config(cfg) -> None:
    """Fail before loading data when the clean training contract is invalid."""
    backend = validate_sdpa_backend(cfg.sdpa_backend)
    if backend != "math":
        raise ValueError(
            "INTACT training is standardized on Math SDPA; "
            f"received sdpa_backend={backend!r}"
        )
    if int(cfg.num_frames) < 3:
        raise ValueError("Actor training requires num_frames >= 3")
    if not 2 <= int(cfg.history_size) < int(cfg.num_frames):
        raise ValueError("history_size must be in [2, num_frames)")
    if not 0.0 < float(cfg.train_split) < 1.0:
        raise ValueError("train_split must be strictly between 0 and 1")
    if cfg.model.intent_mode not in {"waypoint", "goal_displacement"}:
        raise ValueError(f"Unknown intent mode: {cfg.model.intent_mode}")
    if int(cfg.loader.num_workers) == 0 and bool(cfg.loader.persistent_workers):
        raise ValueError("persistent_workers requires loader.num_workers > 0")
    if int(cfg.log_interval_steps) < 1:
        raise ValueError("log_interval_steps must be positive")
    weights = (
        float(cfg.loss.forward_weight),
        float(cfg.loss.sigreg.weight),
        float(cfg.loss.intent.local_weight),
        float(cfg.loss.intent.goal_weight),
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("Training loss weights must be non-negative")


def build_dataset(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR")
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    action_mean, action_std = get_column_stats(dataset, "action")
    transforms = [get_img_preprocessor("pixels", "pixels", img_size=cfg.img_size)]
    transforms.append(
        get_column_normalizer(
            dataset,
            "action",
            "action",
            stats=(action_mean, action_std),
            raw_nan_value=0.0,
        )
    )

    with open_dict(cfg):
        action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        cfg.model.action_encoder.input_dim = action_dim
        if cfg.model.intent_actor is not None:
            cfg.model.intent_actor.action_dim = action_dim
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return PreviousActionDataset(dataset, action_mean, action_std)


def build_dataloaders(cfg, dataset):
    """Create deterministic train/validation loaders for one task."""
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, validation_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, validation_loader


def build_model(cfg):
    """Instantiate the configured model and optionally load strict weights."""
    model = hydra.utils.instantiate(cfg.model)
    if cfg.init_weights_path:
        state = torch.load(cfg.init_weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
    return model


def constant_lr_metadata(learning_rate: float) -> dict[str, float | str]:
    """Describe the fixed learning-rate contract used by formal training."""
    return {
        "type": "constant",
        "learning_rate": float(learning_rate),
    }


def build_training_module(cfg, model):
    """Build the shared loss/optimization harness used by every train mode."""
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "ConstantLR", "factor": 1.0, "total_iters": 1},
            "interval": "step",
        }
    }
    return spt.Module(
        model=model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(intact_forward, cfg=cfg),
        optim=optimizers,
    )


def current_git_commit() -> str:
    """Return the exact source revision without requiring a GitPython dependency."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_run_artifacts(cfg, *, training_mode: str = "single_task"):
    """Persist the resolved configuration and auditable run metadata."""
    checkpoint_root = Path(
        swm.data.utils.get_cache_dir(sub_folder="checkpoints")
    )
    run_dir = checkpoint_root / cfg.output_model_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains artifacts: {run_dir}. "
            "Choose a new output_model_name; existing runs are never overwritten."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "train_config.yaml")
    metadata = {
        "method": "INTACT",
        "training_mode": training_mode,
        "intent_mode": str(cfg.model.intent_mode),
        "seed": int(cfg.seed),
        "task": str(cfg.data.task_name),
        "dataset": str(cfg.data.dataset.name),
        "action_dim": int(cfg.model.intent_actor.action_dim),
        "num_frames": int(cfg.num_frames),
        "history_size": int(cfg.history_size),
        "max_epochs": int(cfg.trainer.max_epochs),
        "log_interval_steps": int(cfg.log_interval_steps),
        "metrics_jsonl": str(run_dir / "metrics" / "train.jsonl"),
        "optimizer": {
            "type": str(cfg.optimizer.type),
            "learning_rate": float(cfg.optimizer.lr),
            "weight_decay": float(cfg.optimizer.weight_decay),
            "fused": bool(cfg.optimizer.get("fused", False)),
            "scheduler": constant_lr_metadata(cfg.optimizer.lr),
        },
        "forward_supervision": "all_adjacent",
        "intent_supervision": "physical_0_6_goal_0_6",
        "physical_transition_count": int(cfg.num_frames) - 1,
        "goal_transition_count": int(cfg.num_frames) - 1,
        "previous_action_contract": "boundary_aware_raw_zero_left_pad_v1",
        "git_commit": current_git_commit(),
        "hostname": socket.gethostname(),
        "sdpa": sdpa_policy_metadata(cfg.sdpa_backend),
        "status": "configured",
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return run_dir, metadata


def build_trainer(cfg, run_dir, logger):
    """Construct Lightning with the repository's explicit checkpoint policy."""
    return pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg.model,
                epoch_interval=1,
            ),
            RunMetadataCallback(run_dir / "run_metadata.json"),
            TrainingProgressCallback(
                run_dir / "metrics" / "train.jsonl",
                interval_steps=int(cfg.log_interval_steps),
                task=str(cfg.data.task_name),
            ),
        ],
        logger=logger,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        default_root_dir=run_dir,
    )


@hydra.main(
    version_base=None, config_path="./config/train", config_name="intact_goal"
)
def run(cfg):
    validate_training_config(cfg)
    pl.seed_everything(cfg.seed, workers=True)
    dataset = build_dataset(cfg)
    train_loader, validation_loader = build_dataloaders(cfg, dataset)
    model = build_model(cfg)
    module = build_training_module(cfg, model)
    data_module = spt.data.DataModule(
        train=train_loader, val=validation_loader
    )
    run_dir, _ = prepare_run_artifacts(cfg)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer = build_trainer(cfg, run_dir, logger)
    with sdpa_kernel_context(cfg.sdpa_backend):
        trainer.fit(model=module, datamodule=data_module)


if __name__ == "__main__":
    run()
