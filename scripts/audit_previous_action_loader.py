#!/usr/bin/env python3
"""Smoke-test boundary-aware previous-action loading on a real dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="pusht", choices=["pusht", "ogb", "dmc", "tworoom"])
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    config_dir = REPO_ROOT / "config" / "train"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        overrides = [f"data={args.data}"]
        if args.dataset_path is not None:
            overrides.append(f"data.dataset.name={args.dataset_path.resolve()}")
        cfg = compose(config_name="intact_goal", overrides=overrides)
    dataset = build_dataset(cfg)
    indices = sorted(
        {
            0,
            1,
            int(dataset.frameskip),
            int(dataset.frameskip) * 2,
            len(dataset) - 1,
        }
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=len(indices),
        num_workers=args.num_workers,
        persistent_workers=False,
    )
    batch = next(iter(loader))
    action = batch["action"]
    previous = batch["previous_action"]
    assert torch.isfinite(action).all()
    assert torch.isfinite(previous).all()
    assert torch.equal(previous[:, 1:], action[:, :-1])
    print(
        {
            "data": args.data,
            "action_shape": tuple(action.shape),
            "previous_action_shape": tuple(previous.shape),
            "num_workers": args.num_workers,
            "status": "ok",
        }
    )


if __name__ == "__main__":
    main()
