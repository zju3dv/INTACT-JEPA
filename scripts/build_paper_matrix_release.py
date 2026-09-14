#!/usr/bin/env python3
"""Build the six-cell E5 checkpoint release from the audited experiment tree."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


SEEDS = (0, 42, 3072)
TASKS = ("pusht", "cube", "reacher", "tworoom")
CORE_ROOT = Path("experiments/recovery_INTACT/results/h203_eval_staging")
GOAL_ROOT = Path(
    "experiments/recovery_INTACT/actor_delta_followup/results/eval_staging"
)

CELLS = (
    {
        "id": "lewm",
        "name": "LeWM",
        "root": CORE_ROOT,
        "run": "lewm",
        "checkpoint": "recovery_lewm",
        "asset": "paper-e5-lewm",
        "native_interface": "CEM 300x30",
        "official_sr": [74.56, 67.33, 83.11, 39.67, 66.17],
        "official_std": [3.67, 1.86, 0.96, 8.39, 2.67],
    },
    {
        "id": "inverse_only",
        "name": "Inverse only",
        "root": CORE_ROOT,
        "run": "inverse_only",
        "checkpoint": "recovery_inverse_only",
        "asset": "paper-e5-inverse-only",
        "native_interface": "Direct",
        "official_sr": [36.11, 67.56, 90.56, 78.56, 68.19],
        "official_std": [0.19, 4.03, 1.35, 6.68, 0.67],
    },
    {
        "id": "waypoint_intent",
        "name": "Waypoint intent only",
        "root": CORE_ROOT,
        "run": "query_only",
        "checkpoint": "recovery_query_only",
        "asset": "paper-e5-waypoint-intent-only",
        "native_interface": "Direct",
        "official_sr": [58.89, 100.00, 65.89, 72.11, 74.22],
        "official_std": [0.69, 0.00, 6.71, 9.83, 3.23],
    },
    {
        "id": "goal_intent",
        "name": "Goal intent only",
        "root": GOAL_ROOT,
        "run": "delta_condition_query_only",
        "checkpoint": "recovery_delta_query_only",
        "asset": "paper-e5-goal-intent-only",
        "native_interface": "Direct",
        "official_sr": [81.78, 100.00, 88.67, 69.22, 84.92],
        "official_std": [0.96, 0.00, 0.33, 7.04, 1.86],
    },
    {
        "id": "waypoint_intact",
        "name": "Waypoint INTACT",
        "root": CORE_ROOT,
        "run": "full",
        "checkpoint": "recovery_full",
        "asset": "paper-e5-waypoint-intact",
        "native_interface": "Direct",
        "official_sr": [71.22, 99.00, 58.22, 77.22, 76.42],
        "official_std": [3.36, 0.67, 9.10, 2.52, 2.32],
    },
    {
        "id": "goal_intact",
        "name": "Goal-displacement INTACT",
        "root": GOAL_ROOT,
        "run": "delta_condition_full",
        "checkpoint": "recovery_delta_full",
        "asset": "intact-goal-e5",
        "native_interface": "Direct",
        "official_sr": [86.11, 100.00, 97.22, 81.56, 91.22],
        "official_std": [0.96, 0.00, 0.84, 3.67, 0.51],
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_dir(source_root: Path, cell: dict, seed: int, task: str) -> Path:
    return (
        source_root
        / cell["root"]
        / f"{cell['run']}_s{seed}"
        / "epoch_5/cache/checkpoints"
        / f"{cell['checkpoint']}_{task}_s{seed}"
    )


def inspect_seed(source_root: Path, cell: dict, seed: int) -> dict:
    shards = []
    shared_hashes = set()
    for task in TASKS:
        directory = checkpoint_dir(source_root, cell, seed, task)
        required = (
            directory / "config.json",
            directory / "config.yaml",
            directory / "multitask_metadata_epoch_5.json",
            directory / "weights_epoch_5.pt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing release files: {missing}")
        metadata = json.loads(required[2].read_text())
        expected = {"epoch": 5, "seed": seed, "task": task}
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{directory}: metadata mismatch {mismatches}")
        shared_hashes.add(metadata["shared_state_sha256"])
        weights = required[-1]
        shards.append(
            {
                "task": task,
                "path": str(Path("checkpoints") / directory.name / weights.name),
                "bytes": weights.stat().st_size,
                "sha256": sha256(weights),
                "config_sha256": sha256(required[0]),
            }
        )
    if len(shared_hashes) != 1:
        raise ValueError(
            f"{cell['id']} seed {seed}: task shards do not share one encoder state"
        )
    return {
        "seed": seed,
        "asset": f"{cell['asset']}-seed{seed}.tar.gz",
        "shared_state_sha256": shared_hashes.pop(),
        "shards": shards,
    }


def build_archive(
    source_root: Path,
    stage_assets: Path,
    cell: dict,
    record: dict,
    force: bool,
    pigz_threads: int,
) -> tuple[str, int, str]:
    archive = stage_assets / record["asset"]
    if force or not archive.is_file():
        seed = record["seed"]
        cache_root = (
            source_root
            / cell["root"]
            / f"{cell['run']}_s{seed}"
            / "epoch_5/cache"
        )
        members = [str(Path(shard["path"]).parent) for shard in record["shards"]]
        command = [
            "tar",
            "-I",
            f"pigz -p {pigz_threads}",
            "-cf",
            str(archive),
            "-C",
            str(cache_root),
            *members,
        ]
        subprocess.run(command, check=True)
    return archive.name, archive.stat().st_size, sha256(archive)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--pigz-threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if shutil.which("pigz") is None:
        raise RuntimeError("pigz is required to build the release archives")
    assets = args.stage_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "release": {
            "provider": "huggingface",
            "repository": "INTACT-JEPA/INTACT",
            "revision": "paper-e5-goal-v1",
        },
        "protocol": {
            "name": "Controlled Math-SDPA shared-encoder E5 matrix",
            "result_revision": "2026-09-14",
            "evaluation_protocol": "official_expert_continuation_v1",
            "epoch": 5,
            "training_seeds": list(SEEDS),
            "evaluation_seeds": [0, 1, 42],
            "episodes_per_evaluation_seed": 100,
            "tasks": list(TASKS),
            "history_boundary": (
                "Use the causal expert action prefix at a sampled continuation; "
                "left-pad only a true episode boundary with normalized raw-zero actions."
            ),
        },
        "cells": [],
    }
    archive_jobs = []
    for cell in CELLS:
        names = ("pusht", "cube", "reacher", "tworoom", "macro")
        aggregate = {
            name: {"mean": mean, "sample_std": std}
            for name, mean, std in zip(
                names, cell["official_sr"], cell["official_std"], strict=True
            )
        }
        records = [inspect_seed(args.source_root, cell, seed) for seed in SEEDS]
        manifest["cells"].append(
            {
                "id": cell["id"],
                "name": cell["name"],
                "native_interface": cell["native_interface"],
                "aggregate_official_sr_percent": aggregate,
                "training_seeds": records,
            }
        )
        archive_jobs.extend((cell, record) for record in records)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                build_archive,
                args.source_root,
                assets,
                cell,
                record,
                args.force,
                args.pigz_threads,
            )
            for cell, record in archive_jobs
        ]
        archive_rows = [future.result() for future in futures]

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2) + "\n"
    args.manifest.write_text(payload)
    (assets / args.manifest.name).write_text(payload)
    checksum_lines = [f"{digest}  {name}" for name, _, digest in archive_rows]
    (assets / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    print(f"Built {len(archive_rows)} archives and {args.manifest}")
    for name, size, digest in archive_rows:
        print(f"{digest}  {size:>10}  {name}")


if __name__ == "__main__":
    main()
