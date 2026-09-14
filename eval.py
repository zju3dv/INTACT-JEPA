import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    if visible_device.isdigit():
        os.environ["MUJOCO_EGL_DEVICE_ID"] = visible_device

import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from evaluation_contract import (
    audit_checkpoint,
    build_evaluation_contract,
    validate_guarded_a_config,
)
from history_policy import (
    BlockStandardScaler,
    StatefulActionHistoryPolicy,
    build_initial_history,
)
from sdpa_policy import (
    sdpa_kernel_context,
    sdpa_policy_metadata,
    validate_sdpa_backend,
)
from utils import load_episode_split_manifest, select_episode_rows


def validate_evaluation_contract(cfg: DictConfig) -> dict[str, str]:
    protocol = str(OmegaConf.select(cfg, "eval.protocol", default="official"))
    if protocol != "official":
        raise ValueError(
            "eval.py implements the official protocol; use clear_eval.py for "
            "CLEAR-LeWM v0.8"
        )
    mode = str(OmegaConf.select(cfg, "eval.inference_mode", default="direct"))
    target = str(OmegaConf.select(cfg, "solver._target_"))
    contract = build_evaluation_contract(protocol, mode, target)
    if mode == "guarded_a":
        validate_guarded_a_config(
            {
                "num_samples": int(cfg.solver.num_samples),
                "n_steps": int(cfg.solver.n_steps),
                "topk": int(cfg.solver.topk),
                "horizon": int(cfg.plan_config.horizon),
                "receding_horizon": int(cfg.plan_config.receding_horizon),
                "action_block": int(cfg.plan_config.action_block),
                "var_scale": float(cfg.solver.var_scale),
                "update_alpha": float(cfg.solver.update_alpha),
                "std_floor": float(cfg.solver.std_floor),
                "std_cap": float(cfg.solver.std_cap),
                "actor_covariance": bool(cfg.solver.actor_covariance),
                "trust_lambda": float(cfg.solver.trust_lambda),
            }
        )
    return contract


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_runtime_sources(mode: str) -> dict[str, str]:
    """Fingerprint the exact local adapter and solver used by an evaluation."""
    root = Path(__file__).resolve().parent
    solver_file = {
        "direct": "direct_solver.py",
        "pure_cem": "cem_solvers.py",
        "actor_cem": "cem_solvers.py",
        "guarded_a": "abcd_solvers.py",
    }[mode]
    names = ["eval.py", "evaluation_contract.py", solver_file]
    if mode != "pure_cem":
        names.append("history_policy.py")
    return {name: _sha256_file(root / name) for name in names}


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Evaluate one explicit INTACT inference mode with the official protocol."""
    evaluation_contract = validate_evaluation_contract(cfg)
    sdpa_backend = validate_sdpa_backend(
        OmegaConf.select(cfg, "sdpa_backend", default="auto")
    )
    protocol = evaluation_contract["protocol"]
    inference_mode = evaluation_contract["inference_mode"]
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    episode_manifest_path = OmegaConf.select(
        cfg, "eval.episode_manifest", default=None
    )
    split_manifest = None
    stats_episode_ids = None
    if episode_manifest_path:
        split_manifest = load_episode_split_manifest(
            episode_manifest_path, dataset=dataset
        )
        stats_episode_ids = split_manifest["split_arrays"]["train"]
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        if stats_episode_ids is not None:
            col_data = select_episode_rows(
                stats_dataset, col_data, stats_episode_ids
            )
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")
    solver = None

    if policy != "random":
        checkpoint_audit = audit_checkpoint(
            cfg.policy,
            swm.data.utils.get_cache_dir(sub_folder="checkpoints"),
        )
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        checkpoint_audit = None
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    g = np.random.default_rng(cfg.seed)
    if split_manifest is not None:
        requested_split = str(
            OmegaConf.select(cfg, "eval.episode_split", default="test")
        )
        if requested_split not in split_manifest["split_arrays"]:
            raise ValueError(f"Unknown evaluation episode split: {requested_split}")
        candidate_episodes = split_manifest["split_arrays"][requested_split]
        max_start_idx = (
            np.asarray(dataset.lengths)[candidate_episodes]
            - cfg.eval.goal_offset_steps
            - 1
        )
        valid = max_start_idx >= 0
        candidate_episodes = candidate_episodes[valid]
        max_start_idx = max_start_idx[valid]
        if candidate_episodes.size < cfg.eval.num_eval:
            raise ValueError(
                f"Split {requested_split} has only {candidate_episodes.size} "
                f"valid episodes for {cfg.eval.num_eval} evaluations"
            )
        selected_positions = g.choice(
            candidate_episodes.size,
            size=cfg.eval.num_eval,
            replace=False,
        )
        eval_episodes = candidate_episodes[selected_positions]
        selected_max_starts = max_start_idx[selected_positions]
        eval_start_idx = np.asarray(
            [g.integers(0, int(value) + 1) for value in selected_max_starts],
            dtype=np.int64,
        )
        order = np.argsort(eval_episodes)
        eval_episodes = eval_episodes[order]
        eval_start_idx = eval_start_idx[order]
        print(
            "EPISODE_DISJOINT_EVAL="
            + json.dumps(
                {
                    "manifest": split_manifest["path"],
                    "split": requested_split,
                    "episodes": eval_episodes.tolist(),
                    "start_steps": eval_start_idx.tolist(),
                },
                sort_keys=True,
            )
        )
    else:
        episode_len = get_episodes_length(dataset, ep_indices)
        max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
        max_start_idx_dict = {
            ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
        }
        max_start_per_row = np.array(
            [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
        )
        valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]
        print(valid_mask.sum(), "valid starting points found for evaluation.")
        random_episode_indices = g.choice(
            len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
        )
        random_episode_indices = np.sort(valid_indices[random_episode_indices])
        print(random_episode_indices)
        eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
        eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    history_protocol = None
    if cfg.policy != "random" and inference_mode != "pure_cem":
        history_slots = int(cfg.plan_config.action_block)
        initial_history = build_initial_history(
            dataset,
            eval_episodes.tolist(),
            eval_start_idx.tolist(),
            history_slots,
        )
        policy.process["action"] = BlockStandardScaler(policy.process["action"])
        policy = StatefulActionHistoryPolicy(policy, initial_history)
        history_protocol = {
            "mode": "expert_continuation",
            "slots": history_slots,
            "initial_rows": "rows[t-slots:t]",
            "padding": "raw-zero left padding only before episode start",
            "online_update": "shift in executed primitive actions",
            "current_dataset_row_action_used": False,
            "target_action_visible": False,
        }
        print("EVAL_ACTION_HISTORY=" + json.dumps(history_protocol, sort_keys=True))

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    with sdpa_kernel_context(sdpa_backend):
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get("callables"), resolve=True
            ),
            video=results_path,
        )
    end_time = time.time()
    eval_total_time = end_time - start_time
    if hasattr(solver, "timing_summary"):
        timing = solver.timing_summary()
        if isinstance(metrics, dict):
            metrics["solver_timing"] = timing
            metrics["eval_total_time"] = eval_total_time
            metrics["cem_time_per_step"] = timing.get("solve_time_mean", 0.0)
            metrics["cem_time_per_episode"] = (
                timing.get("solve_time_sum", 0.0) / max(int(cfg.eval.num_eval), 1)
            )
            metrics["get_cost_calls"] = timing.get("get_cost_calls_sum", 0.0)
            metrics["candidate_action_sequences"] = timing.get(
                "candidate_action_sequences_sum", 0.0
            )
            metrics["candidate_action_steps"] = timing.get(
                "candidate_action_steps_sum", 0.0
            )
            metrics["configured_candidate_sequences_per_solve"] = timing.get(
                "configured_candidate_sequences_per_solve_mean", 0.0
            )
            metrics["configured_candidate_action_steps_per_solve"] = timing.get(
                "configured_candidate_action_steps_per_solve_mean", 0.0
            )
            metrics["final_mean_rescored_sequences"] = timing.get(
                "final_mean_rescored_sequences_sum", 0.0
            )

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = results_path.with_suffix(results_path.suffix + ".json")

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {eval_total_time} seconds\n")

    resolved_config = OmegaConf.to_yaml(cfg, resolve=True)
    runtime = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_index": (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        ),
        "cuda_device_name": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        ),
    }
    payload = {
        "policy": cfg.policy,
        "protocol": protocol,
        "inference_mode": inference_mode,
        "evaluation_contract": evaluation_contract,
        "action_history_protocol": history_protocol,
        "checkpoint_audit": checkpoint_audit,
        "evaluation_seed": int(cfg.seed),
        "evaluation_config_sha256": hashlib.sha256(
            resolved_config.encode("utf-8")
        ).hexdigest(),
        "sdpa": sdpa_policy_metadata(sdpa_backend),
        "runtime": runtime,
        "runtime_source_sha256": audit_runtime_sources(inference_mode),
        "num_eval": cfg.eval.num_eval,
        "num_samples": OmegaConf.select(cfg, "solver.num_samples", default=None),
        "n_steps": OmegaConf.select(cfg, "solver.n_steps", default=None),
        "topk": OmegaConf.select(cfg, "solver.topk", default=None),
        "initial_action_std": OmegaConf.select(
            cfg, "solver.var_scale", default=None
        ),
        "horizon": cfg.plan_config.horizon,
        "receding_horizon": cfg.plan_config.receding_horizon,
        "action_block": cfg.plan_config.action_block,
        "metrics": _json_safe(metrics),
        "eval_total_time": eval_total_time,
        "output_file": str(results_path),
        "episode_split_manifest": (
            split_manifest["path"] if split_manifest is not None else None
        ),
        "episode_split": (
            requested_split if split_manifest is not None else None
        ),
        "eval_episodes": _json_safe(eval_episodes),
        "eval_start_steps": _json_safe(eval_start_idx),
    }
    with sidecar_path.open("w") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    run()
