#!/usr/bin/env python3
"""Fast dependency and environment-registration check for public installs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import torch

import stable_pretraining  # noqa: F401
import stable_worldmodel  # noqa: F401

from direct_solver import DirectSolver  # noqa: F401
from jepa import JEPA  # noqa: F401
from module import IntentActionActor, SIGReg  # noqa: F401


PACKAGES = (
    "stable-worldmodel",
    "stable-pretraining",
    "lightning",
    "hydra-core",
    "ogbench",
    "mujoco",
    "pylance",
)
ENV_IDS = (
    "swm/PushT-v1",
    "swm/OGBCube-v0",
    "swm/ReacherDMControl-v0",
    "swm/TwoRoom-v1",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail when CUDA is unavailable",
    )
    args = parser.parse_args()

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__} (CUDA build: {torch.version.cuda})")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable to PyTorch")

    for package in PACKAGES:
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError as error:
            raise RuntimeError(f"missing package: {package}") from error

    for env_id in ENV_IDS:
        spec = gym.spec(env_id)
        print(f"environment: {env_id} -> {spec.entry_point}")

    print("INTACT install check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
