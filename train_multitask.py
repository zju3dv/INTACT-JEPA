"""Four-task INTACT training with a shared visual representation.

Launch exactly one process per task. Each rank owns one dataset and one set of
Forward/action/Actor heads. Encoder and projector gradients and buffers are
synchronized, matching the task-specific-head structure in Fig. 1 without
mixing action dimensions or normalization statistics.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import socket
import time
from itertools import chain
from pathlib import Path

import stable_worldmodel as swm
import lightning as pl
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from sdpa_policy import sdpa_kernel_context, sdpa_policy_metadata, validate_sdpa_backend
from train import (
    build_dataset,
    build_dataloaders,
    build_model,
    constant_lr_metadata,
    current_git_commit,
    intact_forward,
    validate_training_config,
)


class LossHarness:
    """Minimal owner object required by the shared ``intact_forward`` core."""

    def __init__(self, model, sigreg):
        self.model = model
        self.sigreg = sigreg

    def log_dict(self, *_args, **_kwargs):
        return None


def validate_deterministic_runtime(enabled: bool) -> None:
    """Require the CUDA determinism contract before CUDA is initialized."""
    if not enabled:
        return
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "deterministic_algorithms=true requires CUBLAS_WORKSPACE_CONFIG "
            "to be :4096:8 or :16:8 before Python starts; launch with "
            "scripts/train_multitask.sh"
        )


def destroy_process_group_if_initialized() -> None:
    """Clean up NCCL on both successful exits and Python exceptions."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def validate_multitask_config(cfg) -> tuple[str, ...]:
    """Validate topology before initializing CUDA or loading datasets."""
    tasks = tuple(cfg.tasks.keys())
    if len(tasks) < 2 or len(set(tasks)) != len(tasks):
        raise ValueError("Multitask training requires at least two unique tasks")
    backend = validate_sdpa_backend(cfg.sdpa_backend)
    if backend != "math":
        raise ValueError("Multitask training is standardized on Math SDPA")
    if int(cfg.epochs) < 1:
        raise ValueError("epochs must be positive")
    if cfg.max_steps is not None and int(cfg.max_steps) < 1:
        raise ValueError("max_steps must be positive when provided")
    if int(cfg.batch_size) < 1 or int(cfg.num_workers) < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if not 0.0 < float(cfg.train_split) < 1.0:
        raise ValueError("train_split must be strictly between 0 and 1")
    if int(cfg.teacher_horizon) < 1:
        raise ValueError("teacher_horizon must be positive")
    if float(cfg.sigreg_weight) < 0:
        raise ValueError("sigreg_weight must be non-negative")
    if int(cfg.log_interval_steps) < 1:
        raise ValueError("log_interval_steps must be positive")
    return tasks


def shared_modules(model):
    """Modules shared across visual domains in the Fig. 1 contract."""
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
    for gradient, value in zip(gradients, synchronized, strict=True):
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
        for buffer, value in zip(floating, synchronized, strict=True):
            buffer.copy_(value)
    for buffer in buffers:
        if not buffer.is_floating_point():
            dist.broadcast(buffer, src=0)


def shared_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for prefix, module in zip(
        ("encoder", "projector"), shared_modules(model), strict=True
    ):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(f"{prefix}.{name}".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def scalar_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: float(value.detach())
        for key, value in metrics.items()
        if torch.is_tensor(value) and value.ndim == 0
    }


def compose_task_config(config_dir: Path, base_cfg, task: str):
    data_config = str(base_cfg.tasks[task])
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name=str(base_cfg.single_task_config),
            overrides=[f"data={data_config}"],
        )
    with open_dict(cfg):
        cfg.seed = int(base_cfg.seed)
        cfg.train_split = float(base_cfg.train_split)
        cfg.sdpa_backend = str(base_cfg.sdpa_backend)
        cfg.loader.batch_size = int(base_cfg.batch_size)
        cfg.loader.num_workers = int(base_cfg.num_workers)
        cfg.loader.persistent_workers = int(base_cfg.num_workers) > 0
        if int(base_cfg.num_workers) == 0:
            cfg.loader.pop("prefetch_factor", None)
        cfg.optimizer.lr = float(base_cfg.learning_rate)
        cfg.optimizer.weight_decay = float(base_cfg.weight_decay)
        cfg.loss.sigreg.weight = float(base_cfg.sigreg_weight)
        cfg.trainer.max_epochs = int(base_cfg.epochs)
        cfg.output_model_name = (
            f"{base_cfg.output_model_name}_{task}"
        )
    validate_training_config(cfg)
    teacher_horizon = int(cfg.num_frames) - int(cfg.history_size)
    if teacher_horizon != int(base_cfg.teacher_horizon):
        raise ValueError(
            "Configured N+H window mismatch: "
            f"num_frames-history_size={teacher_horizon}, "
            f"expected {int(base_cfg.teacher_horizon)}"
        )
    return cfg


def save_task_checkpoint(model, cfg, task: str, epoch: int, metadata: dict) -> str:
    swm.wm.utils.save_pretrained(
        model,
        run_name=str(cfg.output_model_name),
        config=cfg.model,
        filename=f"weights_epoch_{epoch}.pt",
    )
    checkpoint_root = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
    checkpoint_dir = checkpoint_root / str(cfg.output_model_name)
    OmegaConf.save(cfg, checkpoint_dir / "train_config.yaml")
    atomic_json(
        checkpoint_dir / f"multitask_metadata_epoch_{epoch}.json",
        {**metadata, "task": task, "checkpoint_dir": str(checkpoint_dir)},
    )
    atomic_json(
        checkpoint_dir / "run_metadata.json",
        {
            **metadata,
            "task": task,
            "checkpoint_dir": str(checkpoint_dir),
            "status": "complete",
        },
    )
    return str(checkpoint_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config/train/intact_multitask.yaml",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-model-name")
    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    with open_dict(cfg):
        for key in ("epochs", "max_steps", "batch_size", "num_workers"):
            value = getattr(args, key)
            if value is not None:
                cfg[key] = value
        if args.output_model_name is not None:
            cfg.output_model_name = args.output_model_name
    return cfg


def main() -> int:
    args = parse_args()
    cfg = apply_cli_overrides(OmegaConf.load(args.config), args)
    tasks = validate_multitask_config(cfg)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != len(tasks):
        raise RuntimeError(
            f"Expected one torchrun process per task ({len(tasks)}), got {world_size}"
        )

    validate_deterministic_runtime(bool(cfg.deterministic_algorithms))

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    task = tasks[rank]
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    atexit.register(destroy_process_group_if_initialized)
    pl.seed_everything(int(cfg.seed), workers=True)
    torch.use_deterministic_algorithms(bool(cfg.deterministic_algorithms))
    torch.backends.cudnn.benchmark = False
    source_commit = current_git_commit()

    task_cfg = compose_task_config(
        Path(__file__).parent / "config/train", cfg, task
    )
    dataset = build_dataset(task_cfg)
    train_loader, _ = build_dataloaders(task_cfg, dataset)
    model = build_model(task_cfg).to(device)
    sigreg = SIGReg(**task_cfg.loss.sigreg.kwargs).to(device)
    harness = LossHarness(model, sigreg)
    broadcast_shared_state(model)

    local_batches = torch.tensor([len(train_loader)], device=device, dtype=torch.long)
    gathered_batches = [torch.zeros_like(local_batches) for _ in tasks]
    dist.all_gather(gathered_batches, local_batches)
    batches_by_task = {
        name: int(value.item())
        for name, value in zip(tasks, gathered_batches, strict=True)
    }
    steps_per_epoch = max(batches_by_task.values())
    if cfg.max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, int(cfg.max_steps))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        fused=True,
    )
    total_steps = steps_per_epoch * int(cfg.epochs)
    shared = shared_parameters(model)
    shared_ids = {id(parameter) for parameter in shared}
    task_specific = [
        parameter for parameter in model.parameters() if id(parameter) not in shared_ids
    ]
    run_dir = (
        Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
        / str(cfg.output_model_name)
    )
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, run_dir / "multitask_config.yaml")
    dist.barrier()
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"rank_{rank}_{task}.jsonl"
    if metrics_path.exists():
        raise FileExistsError(
            f"Refusing to append to an existing metrics file: {metrics_path}"
        )
    metrics_path.touch()
    dist.barrier()

    started = time.time()
    with sdpa_kernel_context(cfg.sdpa_backend):
        for epoch in range(1, int(cfg.epochs) + 1):
            model.train()
            iterator = iter(train_loader)
            cycles = 0
            last_metrics = None
            for step in range(steps_per_epoch):
                try:
                    batch = next(iterator)
                except StopIteration:
                    cycles += 1
                    iterator = iter(train_loader)
                    batch = next(iterator)
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    last_metrics = intact_forward(harness, batch, "train", task_cfg)
                last_metrics["loss"].backward()
                average_shared_gradients(model, world_size)
                torch.nn.utils.clip_grad_norm_(shared, float(cfg.gradient_clip_val))
                torch.nn.utils.clip_grad_norm_(
                    task_specific, float(cfg.gradient_clip_val)
                )
                optimizer.step()
                average_shared_buffers(model, world_size)
                global_step = (epoch - 1) * steps_per_epoch + step + 1
                should_log = (
                    step == 0
                    or (step + 1) % int(cfg.log_interval_steps) == 0
                    or step + 1 == steps_per_epoch
                )
                if should_log:
                    elapsed = time.time() - started
                    steps_per_second = global_step / max(elapsed, 1e-9)
                    record = {
                        "epoch": epoch,
                        "step": step + 1,
                        "steps_per_epoch": steps_per_epoch,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "task": task,
                        "rank": rank,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "steps_per_second": steps_per_second,
                        "eta_seconds": (total_steps - global_step)
                        / max(steps_per_second, 1e-9),
                        **scalar_metrics(last_metrics),
                    }
                    append_jsonl(metrics_path, record)
                    print(
                        "MULTITASK_PROGRESS="
                        + json.dumps(record, sort_keys=True),
                        flush=True,
                    )

            shared_hash = shared_state_sha256(model)
            hashes = [None for _ in tasks]
            dist.all_gather_object(hashes, shared_hash)
            if len(set(hashes)) != 1:
                raise RuntimeError(f"Shared states diverged: {hashes}")
            metadata = {
                "method": "INTACT",
                "training_mode": "multitask_shared_encoder",
                "architecture": (
                    "shared encoder/projector; task-specific predictor, "
                    "action encoder, pred projector, and intent actor"
                ),
                "tasks": list(tasks),
                "action_dim": int(task_cfg.model.intent_actor.action_dim),
                "rank": rank,
                "seed": int(cfg.seed),
                "epoch": epoch,
                "steps_per_epoch": steps_per_epoch,
                "global_steps": epoch * steps_per_epoch,
                "total_configured_steps": total_steps,
                "loader_batches": len(train_loader),
                "loader_cycles": cycles,
                "batches_by_task": batches_by_task,
                "shared_state_sha256": shared_hash,
                "git_commit": source_commit,
                "hostname": socket.gethostname(),
                "sdpa": sdpa_policy_metadata(cfg.sdpa_backend),
                "deterministic_algorithms": bool(cfg.deterministic_algorithms),
                "teacher_horizon": int(cfg.teacher_horizon),
                "optimizer": {
                    "type": "AdamW",
                    "learning_rate": float(cfg.learning_rate),
                    "weight_decay": float(cfg.weight_decay),
                    "fused": True,
                    "scheduler": constant_lr_metadata(cfg.learning_rate),
                },
                "forward_supervision": "all_adjacent",
                "intent_supervision": "physical_0_6_goal_0_6",
                "physical_transition_count": int(task_cfg.num_frames) - 1,
                "goal_transition_count": int(task_cfg.num_frames) - 1,
                "previous_action_contract": "boundary_aware_raw_zero_left_pad_v1",
                "batch_size_per_task": int(cfg.batch_size),
                "loss_weights": {
                    "forward": float(task_cfg.loss.forward_weight),
                    "sigreg": float(task_cfg.loss.sigreg.weight),
                    "local": float(task_cfg.loss.intent.local_weight),
                    "goal": float(task_cfg.loss.intent.goal_weight),
                },
                "loss": scalar_metrics(last_metrics),
                "metrics_jsonl": str(metrics_path),
                "wall_seconds": time.time() - started,
            }
            checkpoint_dir = save_task_checkpoint(
                model, task_cfg, task, epoch, metadata
            )
            rank_record = {**metadata, "checkpoint_dir": checkpoint_dir}
            atomic_json(
                run_dir / f"epoch_{epoch}" / f"rank_{rank}_{task}.json",
                rank_record,
            )
            dist.barrier()
            if rank == 0:
                records = [
                    json.loads(
                        (run_dir / f"epoch_{epoch}" / f"rank_{index}_{name}.json").read_text()
                    )
                    for index, name in enumerate(tasks)
                ]
                atomic_json(
                    run_dir / f"multitask_metadata_epoch_{epoch}.json",
                    {
                        "status": "complete",
                        "epoch": epoch,
                        "tasks": records,
                        "shared_state_sha256": shared_hash,
                        "sdpa": sdpa_policy_metadata(cfg.sdpa_backend),
                    },
                )
                print(
                    "MULTITASK_EPOCH_COMPLETE="
                    + json.dumps(
                        {"epoch": epoch, "shared_state_sha256": shared_hash},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            dist.barrier()

    destroy_process_group_if_initialized()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
