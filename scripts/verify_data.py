#!/usr/bin/env python3
"""Check the public four-task dataset layout without reading full arrays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DATASETS = {
    "pusht": Path("datasets/pusht_expert_train.lance"),
    "cube": Path("datasets/ogbench/cube_single_expert.h5"),
    "reacher": Path("datasets/reacher.h5"),
    "tworoom": Path("datasets/tworoom.h5"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="stable-worldmodel cache (defaults to LOCAL_DATASET_DIR/STABLEWM_HOME)",
    )
    parser.add_argument(
        "--task",
        choices=("all", *DATASETS),
        default="all",
    )
    args = parser.parse_args()

    root = args.root
    if root is None:
        value = os.environ.get("LOCAL_DATASET_DIR") or os.environ.get(
            "STABLEWM_HOME"
        )
        if not value:
            raise SystemExit(
                "Set LOCAL_DATASET_DIR/STABLEWM_HOME or pass --root."
            )
        root = Path(value)
    root = root.expanduser().resolve()

    tasks = DATASETS if args.task == "all" else {args.task: DATASETS[args.task]}
    missing = []
    for task, relative in tasks.items():
        path = root / relative
        if path.exists():
            kind = "directory" if path.is_dir() else "file"
            print(f"{task}: OK ({kind}) {path}")
        else:
            print(f"{task}: MISSING {path}")
            missing.append(task)

    if missing:
        raise SystemExit("Missing datasets: " + ", ".join(missing))
    print("INTACT data layout: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
