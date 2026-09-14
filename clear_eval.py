"""Run INTACT with explicit solvers on CLEAR-LeWM v0.8 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from evaluation_contract import (
    CLEAR_SOLVER_TARGETS,
    GUARDED_A_CONFIG,
    audit_checkpoint,
    build_evaluation_contract,
    validate_guarded_a_config,
)
from history_policy import (
    BlockStandardScaler,
    StatefulActionHistoryPolicy,
    build_initial_history,
)
from sdpa_policy import sdpa_kernel_context, sdpa_policy_metadata

ROOT = Path(__file__).resolve().parent
SOLVER_TARGETS = CLEAR_SOLVER_TARGETS
CLEAR_LEWM_VERSION = "0.8"
CLEAR_LEWM_COMMIT = "ea27f4d6f469242e451ad30c93958fabdc45d129"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("direct", "pure_cem", "guarded_a"),
    )
    parser.add_argument("--sdpa-backend", choices=("auto", "math"), default="math")
    parser.add_argument("--policy-label")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--solver-batch-size", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"))
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument("--allow-modified-stable-worldmodel", action="store_true")
    parser.add_argument("--random-results", type=Path)
    parser.add_argument("--video-dir", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_clear_source(clear_root: Path) -> dict[str, object]:
    """Require the exact CLEAR-LeWM v0.8 Git commit and fingerprint it."""
    clear_root = clear_root.resolve()
    try:
        head = subprocess.run(
            ["git", "-C", str(clear_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(clear_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "CLEAR evaluation requires a Git checkout at exact v0.8 commit "
            f"{CLEAR_LEWM_COMMIT}; source root was {clear_root}"
        ) from exc
    if head != CLEAR_LEWM_COMMIT:
        raise RuntimeError(
            "CLEAR-LeWM source commit mismatch: "
            f"expected {CLEAR_LEWM_COMMIT}, got {head}"
        )
    if status:
        raise RuntimeError(
            "CLEAR-LeWM v0.8 source checkout is modified; evaluation requires "
            f"a clean tree, got status entries: {status}"
        )
    tracked_files = {
        relative: _sha256(clear_root / relative)
        for relative in ("pyproject.toml", "clear_lewm/runner.py")
    }
    return {
        "version": CLEAR_LEWM_VERSION,
        "commit": head,
        "git_dirty": False,
        "git_status": status,
        "tracked_file_sha256": tracked_files,
    }


def audit_adapter_sources(mode: str) -> dict[str, str]:
    """Fingerprint the local CLEAR adapter, history policy, and selected solver."""
    solver_file = SOLVER_TARGETS[mode].split(".", 1)[0] + ".py"
    names = ["clear_eval.py", "evaluation_contract.py", solver_file]
    if mode != "pure_cem":
        names.append("history_policy.py")
    return {name: _sha256(ROOT / name) for name in names}


@contextmanager
def explicit_cem_solver(runner, mode: str):
    """Select a CEM implementation without mutating the checkpoint model."""
    if mode == "direct":
        yield
        return

    original = runner._compose_config

    def compose_with_explicit_solver(*args, **kwargs):
        from omegaconf import open_dict

        cfg = original(*args, **kwargs)
        with open_dict(cfg.solver):
            cfg.solver._target_ = SOLVER_TARGETS[mode]
            if mode == "guarded_a":
                cfg.solver.num_samples = 128
                cfg.solver.var_scale = 0.25
                cfg.solver.n_steps = 3
                cfg.solver.topk = 16
                cfg.solver.update_alpha = 1.0
                cfg.solver.std_floor = 1e-4
                cfg.solver.std_cap = 10.0
                cfg.solver.actor_covariance = False
                cfg.solver.trust_lambda = 0.0
                cfg.plan_config.horizon = 5
                cfg.plan_config.receding_horizon = 5
                cfg.plan_config.action_block = 5
        return cfg

    runner._compose_config = compose_with_explicit_solver
    try:
        yield
    finally:
        runner._compose_config = original


@contextmanager
def causal_history_evaluation(mode: str):
    """Inject the released action-history contract into CLEAR's fixed runner."""
    if mode == "pure_cem":
        yield
        return

    import stable_worldmodel as swm

    original = swm.World.evaluate

    def evaluate_with_history(world, *args, **kwargs):
        dataset = kwargs.get("dataset")
        episodes = kwargs.get("episodes_idx")
        start_steps = kwargs.get("start_steps")
        if dataset is None or episodes is None or start_steps is None:
            raise ValueError(
                "Causal INTACT evaluation requires dataset, episodes_idx, and "
                "start_steps"
            )
        policy = world.policy
        slots = int(policy.cfg.action_block)
        initial_history = build_initial_history(
            dataset, episodes, start_steps, slots
        )
        original_action_scaler = policy.process["action"]
        policy.process["action"] = BlockStandardScaler(original_action_scaler)
        wrapped = StatefulActionHistoryPolicy(policy, initial_history)
        world.policy = wrapped
        try:
            return original(world, *args, **kwargs)
        finally:
            world.policy = policy
            policy.process["action"] = original_action_scaler

    swm.World.evaluate = evaluate_with_history
    try:
        yield
    finally:
        swm.World.evaluate = original


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_clear_result(
    result: dict,
    mode: str,
    *,
    sdpa_backend: str = "math",
    clear_source_audit: dict[str, object] | None = None,
) -> dict:
    """Validate solver identity and attach the clean INTACT result contract."""
    inference = result.setdefault("inference", {})
    solver_target = str(inference.get("solver_target"))
    contract = build_evaluation_contract(
        "clear-lewm-v0.8", mode, solver_target
    )
    inference["mode"] = mode
    inference["actor_input_contract"] = contract["actor_input_contract"]
    inference["search_initialization"] = contract["search_initialization"]
    inference.pop("actor_warmstart_requested", None)
    inference.pop("actor_warmstart_effective", None)
    result["evaluation_protocol"] = contract["protocol"]
    result["evaluation_contract"] = contract
    result["action_history_protocol"] = (
        None
        if mode == "pure_cem"
        else {
            "mode": "expert_continuation",
            "slots": 5,
            "initial_rows": "rows[t-slots:t]",
            "padding": "raw-zero left padding only before episode start",
            "online_update": "shift in executed primitive actions",
            "current_dataset_row_action_used": False,
            "target_action_visible": False,
        }
    )
    if mode == "guarded_a":
        result.setdefault("solver", {}).update(
            {
                "identity": "Guarded A",
                **GUARDED_A_CONFIG,
                "initial_raw_action_std": GUARDED_A_CONFIG["var_scale"],
                "candidate_sequences_per_solve": 384,
                "candidate_action_steps_per_solve": 1920,
                "final_mean_rescores_per_solve": 1,
            }
        )
    result["clear_lewm_source"] = clear_source_audit
    result["sdpa"] = sdpa_policy_metadata(sdpa_backend)
    result["execution_context"] = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    result["runtime_source_sha256"] = audit_adapter_sources(mode)
    return result


def run(args: argparse.Namespace) -> dict:
    clear_root = args.clear_root.resolve()
    if not (clear_root / "clear_lewm" / "runner.py").is_file():
        raise FileNotFoundError(f"Invalid CLEAR-LeWM source root: {clear_root}")
    clear_source_audit = audit_clear_source(clear_root)
    sys.path.insert(0, str(clear_root))
    sys.path.insert(0, str(ROOT))
    from clear_lewm import runner

    inference_mode = "direct" if args.mode == "direct" else "cem"
    if args.mode == "guarded_a":
        requested = {
            "num_samples": args.num_samples,
            "n_steps": args.n_steps,
            "topk": args.topk,
        }
        canonical = {"num_samples": 128, "n_steps": 3, "topk": 16}
        mismatched = {
            key: value
            for key, value in requested.items()
            if value is not None and value != canonical[key]
        }
        if mismatched:
            raise ValueError(
                f"Guarded A has a fixed release configuration: {canonical}; "
                f"got incompatible overrides {mismatched}"
            )
        validate_guarded_a_config(GUARDED_A_CONFIG)
    upstream_dir = args.upstream_dir or clear_root / "third_party" / "le-wm"
    with sdpa_kernel_context(args.sdpa_backend):
        with explicit_cem_solver(runner, args.mode):
            with causal_history_evaluation(args.mode):
                result = runner.evaluate_manifest(
                    manifest_path=args.manifest,
                    policy=args.policy,
                    output=args.output,
                    cache_dir=args.cache_dir,
                    dataset_name=args.dataset_name,
                    dataset_path=args.dataset_path,
                    upstream_dir=upstream_dir,
                    runtime_dir=ROOT,
                    policy_seed=args.policy_seed,
                    num_samples=args.num_samples,
                    n_steps=args.n_steps,
                    topk=args.topk,
                    actor_warmstart=(args.mode != "pure_cem"),
                    inference_mode=inference_mode,
                    random_results=args.random_results,
                    video_dir=args.video_dir,
                    policy_label=args.policy_label,
                    solver_batch_size=args.solver_batch_size,
                    cpu_threads=args.cpu_threads,
                    matmul_precision=args.matmul_precision,
                    strict_checkpoint=args.strict_checkpoint,
                    allow_modified_stable_worldmodel=(
                        args.allow_modified_stable_worldmodel
                    ),
                )

    if args.cache_dir is not None and args.policy != "random":
        result["clean_checkpoint_audit"] = audit_checkpoint(
            args.policy, args.cache_dir / "checkpoints"
        )

    finalize_clear_result(
        result,
        args.mode,
        sdpa_backend=args.sdpa_backend,
        clear_source_audit=clear_source_audit,
    )
    _write_result(args.output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
