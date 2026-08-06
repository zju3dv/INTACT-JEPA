#!/usr/bin/env python3
"""Train INTACT with a shared encoder/projector and task-specific small heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from itertools import chain
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from train import build_dataset, configure_deterministic_math, intact_forward


TASKS = ("pusht", "cube", "reacher", "tworoom")
DATA_CONFIG = {
    "pusht": "pusht",
    "cube": "ogb",
    "reacher": "dmc",
    "tworoom": "tworoom",
}

VARIANTS = {
    "lewm": {
        "config": "lewm",
        "intent": "goal_displacement",
        "local_weight": 0.0,
        "goal_weight": 0.0,
    },
    "inverse": {
        "config": "intact_inverse",
        "intent": "goal_displacement",
        "local_weight": 0.1,
        "goal_weight": 0.0,
    },
    "goal_only": {
        "config": "intact_goal_only",
        "intent": "goal_displacement",
        "local_weight": 0.0,
        "goal_weight": 0.05,
    },
    "displacement": {
        "config": "intact_goal",
        "intent": "goal_displacement",
        "local_weight": 0.1,
        "goal_weight": 0.05,
    },
    "waypoint": {
        "config": "intact_waypoint",
        "intent": "waypoint",
        "local_weight": 0.1,
        "goal_weight": 0.05,
    },
}


class LossHarness:
    """Minimal interface expected by ``intact_forward`` outside Lightning."""

    def __init__(self, model, sigreg):
        self.model = model
        self.sigreg = sigreg

    def log_dict(self, *_args, **_kwargs):
        return None


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def shared_modules(model):
    return model.encoder, model.projector


def shared_parameters(model):
    return list(
        chain.from_iterable(module.parameters() for module in shared_modules(model))
    )


def shared_buffers(model):
    return list(chain.from_iterable(module.buffers() for module in shared_modules(model)))


@torch.no_grad()
def broadcast_shared_state(model, source: int = 0) -> None:
    for tensor in chain(shared_parameters(model), shared_buffers(model)):
        dist.broadcast(tensor.data, src=source)


def average_shared_gradients(model, world_size: int) -> None:
    parameters = shared_parameters(model)
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradients.append(parameter.grad)
    flat = torch._utils._flatten_dense_tensors(gradients)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    synchronized = torch._utils._unflatten_dense_tensors(flat, gradients)
    for gradient, value in zip(gradients, synchronized):
        gradient.copy_(value)


@torch.no_grad()
def average_shared_buffers(model, world_size: int) -> None:
    buffers = shared_buffers(model)
    floating = [buffer for buffer in buffers if buffer.is_floating_point()]
    if floating:
        flat = torch._utils._flatten_dense_tensors(floating)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        synchronized = torch._utils._unflatten_dense_tensors(flat, floating)
        for buffer, value in zip(floating, synchronized):
            buffer.copy_(value)
    for buffer in buffers:
        if not buffer.is_floating_point():
            dist.broadcast(buffer, src=0)


def shared_state_hash(model) -> str:
    digest = hashlib.sha256()
    for prefix, module in (("encoder", model.encoder), ("projector", model.projector)):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(f"{prefix}.{name}".encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_task(config_dir: Path, task: str, args):
    variant = VARIANTS[args.variant]
    config_name = variant["config"]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(config_name=config_name, overrides=[f"data={DATA_CONFIG[task]}"])
    with open_dict(cfg):
        cfg.seed = args.seed
        cfg.loss.forward_weight = 1.0
        cfg.variant = args.variant
        cfg.model.intent_mode = variant["intent"]
        cfg.model.actor_warmstart = args.variant != "lewm"
        cfg.loss.intent.local_weight = variant["local_weight"]
        cfg.loss.intent.goal_weight = variant["goal_weight"]
        cfg.loss.intent.local_start = 0
        cfg.loss.intent.goal_start = 0
        cfg.loss.sigreg.weight = 0.03
        cfg.optimizer.lr = args.lr
        cfg.optimizer.weight_decay = args.weight_decay
        cfg.loader.batch_size = args.batch_size
        cfg.loader.num_workers = args.num_workers
        cfg.loader.persistent_workers = args.num_workers > 0
        if args.num_workers == 0:
            cfg.loader.pop("prefetch_factor", None)
    return cfg


def build_loader(cfg, seed: int, rank: int):
    dataset = build_dataset(cfg)
    split_generator = torch.Generator().manual_seed(seed)
    train_set, _ = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1.0 - cfg.train_split],
        generator=split_generator,
    )
    loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed + 1000 * rank),
    )
    return loader, {
        "dataset": str(cfg.data.dataset.name),
        "dataset_clips": len(dataset),
        "train_clips": len(train_set),
        "loader_batches": len(loader),
        "data_fraction": 1.0,
        "train_split": float(cfg.train_split),
    }


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def save_task_checkpoint(model, cfg, task: str, epoch: int, args, metrics):
    policy = f"{args.output_prefix}_{task}_s{args.seed}"
    os.environ["STABLEWM_HOME"] = str(args.output_cache)
    swm.wm.utils.save_pretrained(
        model,
        run_name=policy,
        config=cfg.model,
        filename=f"weights_epoch_{epoch}.pt",
    )
    checkpoint_dir = args.output_cache / "checkpoints" / policy
    OmegaConf.save(cfg.model, checkpoint_dir / "config.yaml")
    atomic_json(
        checkpoint_dir / f"training_metadata_epoch_{epoch}.json",
        {
            "task": task,
            "seed": args.seed,
            "epoch": epoch,
            "variant": args.variant,
            "intent": VARIANTS[args.variant]["intent"],
            "shared": ["encoder", "projector"],
            "task_specific": [
                "predictor",
                "action_encoder",
                "pred_proj",
                *([] if model.intent_actor is None else ["intent_actor"]),
            ],
            "loss_weights": {
                "forward": 1.0,
                "inverse": VARIANTS[args.variant]["local_weight"],
                "goal": VARIANTS[args.variant]["goal_weight"],
                "sigreg": 0.03,
            },
            "window_protocol": {
                "local_start": 0,
                "goal_start": 0,
                "terminal_index": 7,
                "local_terms": 7 if VARIANTS[args.variant]["local_weight"] > 0 else 0,
                "goal_terms": 7 if VARIANTS[args.variant]["goal_weight"] > 0 else 0,
            },
            "sdpa": "MATH",
            "metrics": metrics,
        },
    )
    return policy, checkpoint_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="displacement")
    parser.add_argument("--seed", type=int, choices=(0, 42, 3072), default=3072)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-prefix", default="intact_multitask_goal")
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != len(TASKS):
        raise RuntimeError(f"Expected four torchrun processes, got {world_size}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    task = TASKS[rank]
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    configure_deterministic_math(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dist.init_process_group("nccl")

    args.output_cache = args.output_cache.resolve()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path(__file__).resolve().parent / "config" / "train"
    cfg = configure_task(config_dir, task, args)
    loader, data_identity = build_loader(cfg, args.seed, rank)
    model = hydra.utils.instantiate(cfg.model).to(device)
    harness = LossHarness(model, SIGReg(**cfg.loss.sigreg.kwargs).to(device))
    broadcast_shared_state(model)

    identities = [None for _ in TASKS]
    dist.all_gather_object(
        identities,
        {"rank": rank, "task": task, "device": torch.cuda.get_device_name(device), **data_identity},
    )
    batch_counts = [None for _ in TASKS]
    dist.all_gather_object(batch_counts, len(loader))
    total_steps = max(int(value) for value in batch_counts)
    if rank == 0:
        atomic_json(
            args.run_dir / "run_identity.json",
            {
                "method": "INTACT",
                "variant": args.variant,
                "protocol": "paper_multitask",
                "architecture": "shared encoder/projector; task-specific forward/action heads",
                "tasks": identities,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size_per_task": args.batch_size,
                "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
                "loss_weights": {
                    "forward": 1.0,
                    "inverse": VARIANTS[args.variant]["local_weight"],
                    "goal": VARIANTS[args.variant]["goal_weight"],
                    "sigreg": 0.03,
                },
                "window_protocol": {
                    "local_indices": (
                        [0, 6] if VARIANTS[args.variant]["local_weight"] > 0 else []
                    ),
                    "goal_indices": (
                        [0, 6] if VARIANTS[args.variant]["goal_weight"] > 0 else []
                    ),
                    "terminal_index": 7,
                },
                "previous_action_contract": "boundary_aware_raw_zero_left_pad_v1",
                "initial_previous_action": "raw_zero_then_zscore",
                "sdpa": "MATH",
            },
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True
    )
    shared_params = shared_parameters(model)
    shared_ids = {id(parameter) for parameter in shared_params}
    head_params = [parameter for parameter in model.parameters() if id(parameter) not in shared_ids]
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        iterator = iter(loader)
        epoch_started = time.time()
        running = {"loss": 0.0, "pred": 0.0, "local": 0.0, "goal": 0.0, "sigreg": 0.0}
        for step in range(total_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = intact_forward(harness, batch, "train", cfg)
            output["loss"].backward()
            average_shared_gradients(model, world_size)
            torch.nn.utils.clip_grad_norm_(shared_params, args.gradient_clip)
            torch.nn.utils.clip_grad_norm_(head_params, args.gradient_clip)
            optimizer.step()
            average_shared_buffers(model, world_size)

            running["loss"] += float(output["loss"].detach())
            running["pred"] += float(output["pred_loss"].detach())
            running["local"] += float(output["local_nll"].detach())
            running["goal"] += float(output["goal_nll"].detach())
            running["sigreg"] += float(output["sigreg_loss"].detach())
            if (step + 1) % args.log_every == 0 or step + 1 == total_steps:
                denominator = args.log_every if (step + 1) % args.log_every == 0 else (step + 1) % args.log_every
                print(
                    "INTACT_PROGRESS="
                    + json.dumps(
                        {
                            "task": task,
                            "epoch": epoch,
                            "step": step + 1,
                            "total_steps": total_steps,
                            **{key: value / max(denominator, 1) for key, value in running.items()},
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                running = {key: 0.0 for key in running}

        average_shared_buffers(model, world_size)
        shared_hash = shared_state_hash(model)
        hashes = [None for _ in TASKS]
        dist.all_gather_object(hashes, shared_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"Shared states diverged: {hashes}")
        metrics = {
            "epoch_wall_seconds": time.time() - epoch_started,
            "total_wall_seconds": time.time() - started,
            "steps": total_steps,
            "shared_state_sha256": shared_hash,
        }
        policy, checkpoint_dir = save_task_checkpoint(model, cfg, task, epoch, args, metrics)
        atomic_json(
            args.run_dir / "epochs" / f"epoch_{epoch}" / f"rank_{rank}_{task}.json",
            {"task": task, "policy": policy, "checkpoint": str(checkpoint_dir), **metrics},
        )
        dist.barrier()
        if rank == 0:
            atomic_json(
                args.run_dir / "epochs" / f"epoch_{epoch}" / "complete.json",
                {"epoch": epoch, "status": "complete", "shared_state_sha256": shared_hash},
            )

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
