#!/usr/bin/env python3
"""Verify and optionally smoke-load released six-cell paper checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checkpoints" / "PAPER_E5_MATRIX_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_cell(manifest: dict, cell_id: str) -> dict:
    matches = [cell for cell in manifest["cells"] if cell["id"] == cell_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown paper cell: {cell_id}")
    return matches[0]


def selected_records(cell: dict, seeds: set[int]) -> list[dict]:
    records = [row for row in cell["training_seeds"] if row["seed"] in seeds]
    found = {row["seed"] for row in records}
    if found != seeds:
        raise ValueError(f"Unknown training seeds: {sorted(seeds - found)}")
    return records


def load_model(cache_root: Path, policy: str, expected_action_dim: int):
    os.environ["STABLEWM_HOME"] = str(cache_root)
    runtime = str(ROOT / "paper_runtime")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(policy)
    assert model.get_action_dim() == expected_action_dim
    assert model.predict_residual is False
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cell", default="goal_intact")
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    cell = selected_cell(manifest, args.cell)
    records = selected_records(cell, set(args.seed))
    failures = []
    for record in records:
        for shard in record["shards"]:
            path = args.root / shard["path"]
            checkpoint_dir = path.parent
            error = None
            if not path.is_file():
                error = "missing"
            elif path.stat().st_size != shard["bytes"]:
                error = f"size={path.stat().st_size} expected={shard['bytes']}"
            else:
                actual = sha256(path)
                if actual != shard["sha256"]:
                    error = f"sha256={actual} expected={shard['sha256']}"
            companions = [
                checkpoint_dir / "config.json",
                checkpoint_dir / "config.yaml",
                checkpoint_dir / "multitask_metadata_epoch_5.json",
            ]
            missing_companions = [str(item) for item in companions if not item.is_file()]
            if error is None and missing_companions:
                error = f"missing companions={missing_companions}"
            if error is None:
                metadata = json.loads(companions[-1].read_text())
                expected_metadata = {
                    "epoch": 5,
                    "seed": record["seed"],
                    "task": shard["task"],
                    "shared_state_sha256": record["shared_state_sha256"],
                }
                mismatches = {
                    key: (metadata.get(key), value)
                    for key, value in expected_metadata.items()
                    if metadata.get(key) != value
                }
                if mismatches:
                    error = f"metadata mismatches={mismatches}"
            if error is None and args.load:
                policy = str(Path(shard["path"]).relative_to("checkpoints"))
                try:
                    expected_action_dim = 25 if shard["task"] == "cube" else 10
                    model = load_model(args.root, policy, expected_action_dim)
                    del model
                except Exception as exc:  # report all shards before failing
                    error = f"load={exc!r}"
            status = "OK" if error is None else "FAIL"
            print(
                f"{status} cell={cell['id']} seed={record['seed']} "
                f"task={shard['task']} {path}"
            )
            if error:
                print(f"  {error}")
                failures.append((path, error))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
