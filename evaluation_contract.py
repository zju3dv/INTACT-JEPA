"""Shared identity contract for INTACT evaluation entrypoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


OFFICIAL_SOLVER_TARGETS = {
    "direct": "direct_solver.DirectSolver",
    "pure_cem": "cem_solvers.PureCEMSolver",
    "actor_cem": "cem_solvers.ActorCEMSolver",
    "guarded_a": "abcd_solvers.GuardedCEMSolver",
}

CLEAR_SOLVER_TARGETS = {
    "direct": "prior_only_solver.PriorOnlySolver",
    "pure_cem": "cem_solvers.PureCEMSolver",
    "actor_cem": "cem_solvers.ActorCEMSolver",
    "guarded_a": "abcd_solvers.GuardedCEMSolver",
}

PROTOCOL_SOLVER_TARGETS = {
    "official": OFFICIAL_SOLVER_TARGETS,
    "clear-lewm-v0.8": CLEAR_SOLVER_TARGETS,
}

MODE_CONTRACTS = {
    "direct": {
        "actor_input_contract": "fig1-real-previous-action",
        "search_initialization": "none-zero-search",
    },
    "pure_cem": {
        "actor_input_contract": "not-applicable-actor-not-called",
        "search_initialization": "zero-mean-search-distribution",
    },
    "actor_cem": {
        "actor_input_contract": "fig1-real-previous-action",
        "search_initialization": "actor-plan",
    },
    "guarded_a": {
        "actor_input_contract": "fig1-real-previous-action",
        "search_initialization": "bounded-residuals-around-direct-plan",
        "search_budget": "128-samples-x-3-rounds",
        "planning_horizon": "5",
        "receding_horizon": "5",
        "action_block": "5",
        "initial_raw_action_std": "0.25",
        "elite_count": "16",
        "update_alpha": "1.0",
        "std_floor": "0.0001",
        "std_cap": "10.0",
        "actor_covariance": "false",
        "trust_lambda": "0.0",
        "direct_reference_preserved": "true",
        "global_best_preserved": "true",
        "final_mean_rescores": "1",
    },
}


GUARDED_A_CONFIG = {
    "num_samples": 128,
    "n_steps": 3,
    "topk": 16,
    "horizon": 5,
    "receding_horizon": 5,
    "action_block": 5,
    "var_scale": 0.25,
    "update_alpha": 1.0,
    "std_floor": 1e-4,
    "std_cap": 10.0,
    "actor_covariance": False,
    "trust_lambda": 0.0,
}


def validate_guarded_a_config(observed: dict[str, object]) -> None:
    """Reject a run labeled Guarded A unless every canonical field matches."""
    mismatches = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in GUARDED_A_CONFIG.items()
        if observed.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Guarded A configuration mismatch: {mismatches}")


def build_evaluation_contract(
    protocol: str, mode: str, solver_target: str
) -> dict[str, str]:
    """Validate and describe one explicit protocol/mode/solver combination."""
    targets = PROTOCOL_SOLVER_TARGETS.get(protocol)
    if targets is None:
        raise ValueError(f"Unknown evaluation protocol: {protocol}")
    expected = targets.get(mode)
    if expected is None:
        raise ValueError(f"Unknown inference mode: {mode}")
    if solver_target != expected:
        raise ValueError(
            f"Protocol {protocol!r} mode {mode!r} requires solver target "
            f"{expected!r}, got {solver_target!r}"
        )
    return {
        "protocol": protocol,
        "inference_mode": mode,
        "solver_target": solver_target,
        **MODE_CONTRACTS[mode],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def audit_checkpoint(policy: str, checkpoint_root: str | Path) -> dict | None:
    """Return portable hashes and compatibility facts for an Official policy."""
    if policy == "random":
        return None
    root = Path(checkpoint_root).expanduser().resolve()
    candidate = Path(policy).expanduser()
    candidate = candidate.resolve() if candidate.is_absolute() else root / candidate
    if candidate.is_dir():
        weights = sorted(candidate.glob("*.pt"))
        if len(weights) != 1:
            raise ValueError(
                f"Checkpoint directory must contain exactly one .pt file: {candidate}"
            )
        checkpoint = weights[0]
    elif candidate.is_file():
        checkpoint = candidate
    else:
        raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")

    try:
        portable_path = checkpoint.relative_to(root).as_posix()
    except ValueError:
        portable_path = checkpoint.name
    record = {
        "policy_id": policy,
        "runtime_file": portable_path,
        "bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "state_dict_load": "strict",
    }
    config_path = checkpoint.parent / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        record["config_sha256"] = _sha256(config_path)
        record["legacy_ablation_config_detected"] = _contains_key(
            config, "actor_warmstart"
        )
        record["model_target"] = config.get("_target_")
    else:
        record["config_sha256"] = None
        record["legacy_ablation_config_detected"] = None
        record["model_target"] = None
    run_metadata_path = checkpoint.parent / "run_metadata.json"
    if run_metadata_path.is_file():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        record["run_metadata_sha256"] = _sha256(run_metadata_path)
        record["training_seed"] = run_metadata.get("seed")
        record["training_commit"] = run_metadata.get("git_commit")
        record["training_sdpa"] = run_metadata.get("sdpa")
        record["training_status"] = run_metadata.get("status")
    else:
        record["run_metadata_sha256"] = None
        record["training_seed"] = None
        record["training_commit"] = None
        record["training_sdpa"] = None
        record["training_status"] = None
    return record
