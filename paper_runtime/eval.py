import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    if visible_device.isdigit():
        os.environ["MUJOCO_EGL_DEVICE_ID"] = visible_device

import json
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

from utils import load_episode_split_manifest, select_episode_rows


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
    """Run evaluation of dinowm vs random policy."""
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
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        actor_warmstart = bool(OmegaConf.select(cfg, "eval.actor_warmstart", default=True))
        if hasattr(model, "set_actor_warmstart"):
            model.set_actor_warmstart(actor_warmstart)
        elif hasattr(model, "actor_warmstart"):
            model.actor_warmstart = actor_warmstart
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
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

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
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
            metrics["rollout_candidates"] = timing.get(
                "candidate_action_steps_sum", 0.0
            )
            metrics["configured_rollout_candidates_per_solve"] = timing.get(
                "configured_rollout_budget_mean", 0.0
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

    payload = {
        "policy": cfg.policy,
        "actor_warmstart": OmegaConf.select(
            cfg, "eval.actor_warmstart", default=None
        ),
        "solver_batch_size": OmegaConf.select(
            cfg, "solver.batch_size", default=None
        ),
        "num_eval": cfg.eval.num_eval,
        "num_samples": OmegaConf.select(cfg, "solver.num_samples", default=None),
        "n_steps": OmegaConf.select(cfg, "solver.n_steps", default=None),
        "topk": OmegaConf.select(cfg, "solver.topk", default=None),
        "horizon": cfg.plan_config.horizon,
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
