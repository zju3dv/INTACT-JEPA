"""Train the paper-aligned, single-task INTACT model."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

# This must be set before CUDA creates a cuBLAS handle.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import (
    PreviousActionDataset,
    SaveCkptCallback,
    get_column_normalizer,
    get_column_stats,
    get_img_preprocessor,
)


def configure_deterministic_math(seed: int) -> None:
    """Match the deterministic Math-SDPA contract used for paper checkpoints."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pl.seed_everything(seed, workers=True)


def predict_adjacent_latents(model, embeddings, action_embeddings, history_size):
    """Predict every adjacent transition with a bounded history window."""
    transitions = embeddings.size(1) - 1
    if transitions < 1:
        raise ValueError("A training window must contain at least two frames")

    prefix = min(history_size, transitions)
    predictions = [
        model.predict(embeddings[:, :prefix], action_embeddings[:, :prefix])
    ]
    for index in range(prefix, transitions):
        start = max(0, index - history_size + 1)
        prediction = model.predict(
            embeddings[:, start : index + 1],
            action_embeddings[:, start : index + 1],
        )[:, -1:]
        predictions.append(prediction)
    return torch.cat(predictions, dim=1)


def construct_intents(
    embeddings: torch.Tensor,
    intent_mode: str,
    local_start: int = 0,
    goal_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct attached physical intents and deployable terminal-goal intents."""
    max_start = embeddings.size(1) - 2
    if local_start < 0 or local_start > max_start:
        raise ValueError(
            f"local_start must be in [0, T-2], got {local_start} for T={embeddings.size(1)}"
        )
    if goal_start < 0 or goal_start > max_start:
        raise ValueError(
            f"goal_start must be in [0, T-2], got {goal_start} for T={embeddings.size(1)}"
        )
    current = embeddings[:, local_start:-1]
    successor = embeddings[:, local_start + 1 :]
    local = successor - current
    deployment_current = embeddings[:, goal_start:-1]
    goal = embeddings[:, -1:].detach().expand_as(deployment_current)
    displacement = goal - deployment_current
    if intent_mode == "waypoint":
        remaining = torch.arange(
            deployment_current.size(1),
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


def paired_intent_action_loss(model, embeddings, actions, previous_actions, cfg):
    """Apply physical and deployment NLLs through the same action operator."""
    local_weight = float(cfg.loss.intent.local_weight)
    goal_weight = float(cfg.loss.intent.goal_weight)
    local_start = int(cfg.loss.intent.local_start)
    goal_start = int(cfg.loss.intent.goal_start)
    if local_weight <= 0 and goal_weight <= 0:
        zero = embeddings.new_zeros(())
        return {
            "action_loss": zero,
            "local_nll": zero,
            "goal_nll": zero,
            "local_mae": zero,
            "goal_mae": zero,
        }

    local_intent, goal_intent = construct_intents(
        embeddings,
        model.intent_mode,
        local_start=local_start,
        goal_start=goal_start,
    )
    zero = embeddings.new_zeros(())
    local_stats = {"loss": zero, "mae": zero[None]}
    if local_weight > 0:
        local_action = actions[:, local_start : embeddings.size(1) - 1]
        local_previous = previous_actions[:, local_start : embeddings.size(1) - 1]
        local_stats = model.action_nll(
            z=embeddings[:, local_start:-1],
            intent=local_intent,
            previous_action=local_previous,
            target_action=local_action,
        )

    goal_stats = {"loss": zero, "mae": zero[None]}
    if goal_weight > 0:
        goal_action = actions[:, goal_start : embeddings.size(1) - 1]
        goal_previous = previous_actions[:, goal_start : embeddings.size(1) - 1]
        goal_stats = model.action_nll(
            z=embeddings[:, goal_start:-1],
            intent=goal_intent,
            previous_action=goal_previous,
            target_action=goal_action,
        )

    return {
        "action_loss": local_weight * local_stats["loss"]
        + goal_weight * goal_stats["loss"],
        "local_nll": local_stats["loss"],
        "goal_nll": goal_stats["loss"],
        "local_mae": local_stats["mae"].mean().detach(),
        "goal_mae": goal_stats["mae"].mean().detach(),
    }


def intact_forward(self, batch, stage, cfg):
    if "previous_action" not in batch:
        raise KeyError("Boundary-aware previous_action is required")
    if not torch.isfinite(batch["action"]).all():
        raise ValueError("Target actions must be finite")
    if not torch.isfinite(batch["previous_action"]).all():
        raise ValueError("Previous actions must be finite")
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
        batch["action"],
        batch["previous_action"],
        cfg,
    )
    loss = (
        float(cfg.loss.forward_weight) * pred_loss
        + float(cfg.loss.sigreg.weight) * sigreg_loss
        + intent["action_loss"]
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


def build_dataset(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    is_lance = str(dataset_name).endswith(".lance")
    if is_lance:
        # Lance already batches random row access efficiently. Preloading every
        # numeric column duplicates gigabytes once the training stack is imported.
        dataset_cfg["keys_to_cache"] = []
    cache_dir = os.environ.get("LOCAL_DATASET_DIR")
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    action_mean, action_std = get_column_stats(dataset, "action")
    transforms = [
        get_img_preprocessor("pixels", "pixels", img_size=cfg.img_size)
    ]
    # The JEPA consumes pixels and actions only. Proprio/state remain available
    # in the batch for diagnostics but do not need full-column normalization.
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
    if is_lance:
        getattr(dataset, "_cache", {}).pop("action", None)
        update_fetch_columns = getattr(dataset, "_update_fetch_columns", None)
        if update_fetch_columns is not None:
            update_fetch_columns()
    return PreviousActionDataset(dataset, action_mean, action_std)


@hydra.main(
    version_base=None, config_path="./config/train", config_name="intact_goal"
)
def run(cfg):
    configure_deterministic_math(int(cfg.seed))
    dataset = build_dataset(cfg)
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

    model = hydra.utils.instantiate(cfg.model)
    if cfg.init_weights_path:
        state = torch.load(cfg.init_weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }
    module = spt.Module(
        model=model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(intact_forward, cfg=cfg),
        optim=optimizers,
    )
    data_module = spt.data.DataModule(
        train=train_loader, val=validation_loader
    )

    checkpoint_root = Path(
        swm.data.utils.get_cache_dir(sub_folder="checkpoints")
    )
    run_dir = checkpoint_root / cfg.output_model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "train_config.yaml")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "method": "INTACT",
                "variant": str(cfg.variant),
                "objective": f"{cfg.variant}_objective",
                "intent_mode": cfg.model.intent_mode,
                "feature_layout": (
                    cfg.model.intent_actor.feature_layout
                    if cfg.model.intent_actor is not None
                    else "none"
                ),
                "seed": int(cfg.seed),
                "dataset": str(cfg.data.dataset.name),
                "goal_start": int(cfg.loss.intent.goal_start),
                "local_start": int(cfg.loss.intent.local_start),
                "loss_weights": {
                    "forward": float(cfg.loss.forward_weight),
                    "sigreg": float(cfg.loss.sigreg.weight),
                    "local": float(cfg.loss.intent.local_weight),
                    "goal": float(cfg.loss.intent.goal_weight),
                },
                "previous_action_contract": "boundary_aware_raw_zero_left_pad_v1",
                "initial_previous_action": "raw_zero_then_zscore",
                "sdpa": "MATH",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg.model,
                epoch_interval=1,
            )
        ],
        logger=logger,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(model=module, datamodule=data_module)


if __name__ == "__main__":
    run()
