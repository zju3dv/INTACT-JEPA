#!/usr/bin/env python3
"""Validate an INTACT checkout before training or evaluation starts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
INTEGRITY_MANIFEST = REPO / "config" / "code_integrity.json"
CLEAR_VERSION = "0.5.1"
CLEAR_COMMIT = "32f4416c333b5e3147641f70621ce3a9257f5920"
TASKS = ("pusht", "cube", "reacher", "tworoom")
TRAIN_SEEDS = (0, 42, 3072)
EVAL_SEEDS = (0, 1, 42)
TRAIN_DATASETS = {
    "pusht": "datasets/pusht_expert_train.lance",
    "cube": "datasets/ogbench/cube_single_expert.h5",
    "reacher": "datasets/reacher.h5",
    "tworoom": "datasets/tworoom.h5",
}
EVAL_DATASETS = {
    "pusht": "datasets/pusht_expert_train.h5",
    "cube": "datasets/ogbench/cube_single_expert.h5",
    "reacher": "datasets/dmc/reacher_random.h5",
    "tworoom": "datasets/tworoom.h5",
}
REQUIRED_PACKAGES = (
    "torch",
    "stable-worldmodel",
    "stable-pretraining",
    "lightning",
    "hydra-core",
    "omegaconf",
    "numpy",
    "h5py",
)


@dataclass
class Check:
    name: str
    status: str
    detail: str


class Report:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.checks: list[Check] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    def passed(self, name: str, detail: str) -> None:
        self.add(name, "PASS", detail)

    def warn(self, name: str, detail: str) -> None:
        self.add(name, "WARN", detail)

    def fail(self, name: str, detail: str) -> None:
        self.add(name, "FAIL", detail)

    @property
    def ok(self) -> bool:
        return not any(check.status == "FAIL" for check in self.checks)

    def payload(self) -> dict[str, Any]:
        counts = {
            status: sum(check.status == status for check in self.checks)
            for status in ("PASS", "WARN", "FAIL")
        }
        return {
            "schema_version": "intact-preflight-v1",
            "mode": self.mode,
            "status": "PASS" if self.ok else "FAIL",
            "counts": counts,
            "checks": [asdict(check) for check in self.checks],
        }

    def print(self) -> None:
        print(f"INTACT preflight: {self.mode}\n")
        for check in self.checks:
            print(f"[{check.status:4}] {check.name}: {check.detail}")
        counts = self.payload()["counts"]
        print(
            f"\nFINAL: {'PASS' if self.ok else 'FAIL'} "
            f"({counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail)"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args: list[str], cwd: Path = REPO) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def check_git(report: Report, strict_clean: bool) -> None:
    if not (REPO / ".git").exists():
        report.warn("git checkout", "repository metadata is unavailable")
        return
    branch = command(["git", "branch", "--show-current"]).stdout.strip()
    commit = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    if branch == "junhan":
        report.passed("git branch", f"junhan @ {commit[:12]}")
    else:
        report.warn("git branch", f"expected junhan, found {branch or 'detached'} @ {commit[:12]}")
    dirty = command(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.strip()
    if dirty:
        detail = "tracked files have local modifications"
        (report.fail if strict_clean else report.warn)("git worktree", detail)
    else:
        report.passed("git worktree", "tracked files are clean")


def check_integrity(report: Report) -> None:
    if not INTEGRITY_MANIFEST.is_file():
        report.fail("code integrity", f"missing {INTEGRITY_MANIFEST.relative_to(REPO)}")
        return
    try:
        payload = json.loads(INTEGRITY_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        report.fail("code integrity", f"invalid manifest: {exc}")
        return
    mismatches = []
    for relative, expected in payload.get("files", {}).items():
        path = REPO / relative
        if not path.is_file():
            mismatches.append(f"missing:{relative}")
        else:
            actual = sha256(path)
            if actual != expected:
                mismatches.append(f"modified:{relative}")
    if mismatches:
        report.fail("code integrity", ", ".join(mismatches))
    else:
        report.passed(
            "code integrity",
            f"{len(payload.get('files', {}))} canonical files match SHA-256",
        )


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python(report: Report) -> None:
    version = sys.version_info
    if (version.major, version.minor) == (3, 10):
        report.passed("python", f"{version.major}.{version.minor}.{version.micro}")
    else:
        report.warn("python", f"reference is 3.10; current is {version.major}.{version.minor}.{version.micro}")
    missing = []
    versions = []
    for distribution in REQUIRED_PACKAGES:
        installed = package_version(distribution)
        if installed is None:
            missing.append(distribution)
        else:
            versions.append(f"{distribution}={installed}")
    if missing:
        report.fail("dependencies", "missing: " + ", ".join(missing))
    else:
        report.passed("dependencies", ", ".join(versions))


def check_math_contract(report: Report) -> None:
    try:
        import torch
    except ImportError as exc:
        report.fail("PyTorch runtime", str(exc))
        return
    if not torch.cuda.is_available():
        report.fail("CUDA", "torch.cuda.is_available() is false")
        return
    report.passed(
        "CUDA",
        f"torch={torch.__version__}, runtime={torch.version.cuda}, visible_gpus={torch.cuda.device_count()}",
    )
    source = (REPO / "train.py").read_text()
    required = (
        "enable_flash_sdp(False)",
        "enable_mem_efficient_sdp(False)",
        "enable_math_sdp(True)",
        "use_deterministic_algorithms(True",
        "cudnn.deterministic = True",
    )
    missing = [token for token in required if token not in source]
    if missing:
        report.fail("Math-SDPA contract", "missing source guards: " + ", ".join(missing))
    else:
        report.passed("Math-SDPA contract", "deterministic Math-SDPA guards are present")


def check_config_contract(report: Report, mode: str) -> None:
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError:
        report.fail("training config", "OmegaConf is unavailable")
        return
    cfg = OmegaConf.load(REPO / "config/train/intact_goal.yaml")
    model = OmegaConf.load(REPO / "config/train/model/intact.yaml")
    expected = {
        "num_frames": (cfg.num_frames, 8),
        "train_split": (cfg.train_split, 0.9),
        "batch_size": (cfg.loader.batch_size, 256),
        "lr": (cfg.optimizer.lr, 5e-4),
        "weight_decay": (cfg.optimizer.weight_decay, 1e-3),
        "forward_weight": (cfg.loss.forward_weight, 1.0),
        "local_weight": (cfg.loss.intent.local_weight, 0.1),
        "goal_weight": (cfg.loss.intent.goal_weight, 0.05),
        "local_start": (cfg.loss.intent.local_start, 0),
        "goal_start": (cfg.loss.intent.goal_start, 0),
        "feature_layout": (model.intent_actor.feature_layout, "four_slot"),
        "predict_residual": (model.predict_residual, False),
    }
    if mode == "train-single":
        expected["epochs"] = (cfg.trainer.max_epochs, 1)
        expected["sigreg"] = (cfg.loss.sigreg.weight, 0.02)
    mismatches = [
        f"{name}={actual!r} (expected {wanted!r})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        report.fail("training config", "; ".join(mismatches))
    else:
        report.passed("training config", f"{len(expected)} canonical values match")
    variant_expected = {
        "lewm": (0.0, 0.0, None),
        "intact_inverse": (0.1, 0.0, "four_slot"),
        "intact_goal_only": (0.0, 0.05, "four_slot"),
        "intact_goal": (0.1, 0.05, "four_slot"),
        "intact_waypoint": (0.1, 0.05, "four_slot"),
    }
    variant_errors = []
    with initialize_config_dir(
        version_base=None, config_dir=str((REPO / "config/train").resolve())
    ):
        for name, (local, goal, layout) in variant_expected.items():
            variant_cfg = compose(config_name=name)
            actual_layout = (
                None
                if variant_cfg.model.intent_actor is None
                else variant_cfg.model.intent_actor.feature_layout
            )
            actual = (
                float(variant_cfg.loss.intent.local_weight),
                float(variant_cfg.loss.intent.goal_weight),
                actual_layout,
            )
            wanted = (local, goal, layout)
            if actual != wanted:
                variant_errors.append(f"{name}={actual!r}, expected {wanted!r}")
    if variant_errors:
        report.fail("training variant matrix", "; ".join(variant_errors))
    else:
        report.passed("training variant matrix", "five objective variants match")
    if mode == "train-multitask":
        source = (REPO / "train_multitask.py").read_text()
        tokens = (
            'parser.add_argument("--epochs", type=int, default=5)',
            'parser.add_argument("--batch-size", type=int, default=256)',
            "cfg.loss.sigreg.weight = 0.03",
            'parser.add_argument("--variant", choices=tuple(VARIANTS)',
            '"displacement": {',
            '"local_weight": 0.1',
            '"goal_weight": 0.05',
            "torch.optim.AdamW(",
            "fused=True",
            "average_shared_gradients(model, world_size)",
        )
        missing = [token for token in tokens if token not in source]
        if missing:
            report.fail("multi-task contract", "missing: " + ", ".join(missing))
        else:
            report.passed("multi-task contract", "E5, B256, SIGReg 0.03, fused AdamW and shared-gradient averaging")


def resolve_cache(args, report: Report) -> Path | None:
    raw = args.cache_dir or os.environ.get("STABLEWM_HOME") or os.environ.get("LOCAL_DATASET_DIR")
    if not raw:
        report.fail("cache root", "set STABLEWM_HOME or pass --cache-dir")
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        report.fail("cache root", f"directory does not exist: {path}")
        return None
    report.passed("cache root", str(path))
    return path


def validate_dataset(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing: {path}"
    if path.is_dir():
        if not any(path.iterdir()):
            return False, f"empty dataset directory: {path}"
        return True, str(path)
    if path.stat().st_size == 0:
        return False, f"empty dataset file: {path}"
    if path.suffix in {".h5", ".hdf5"}:
        try:
            import h5py

            with h5py.File(path, "r") as handle:
                keys = list(handle.keys())
            if not keys:
                return False, f"HDF5 has no root keys: {path}"
            return True, f"{path} (HDF5 keys={len(keys)})"
        except Exception as exc:
            return False, f"cannot open HDF5 {path}: {exc}"
    return True, str(path)


def check_datasets(
    report: Report,
    mode: str,
    task: str | None,
    cache: Path | None,
    dataset_path: Path | None,
) -> None:
    if cache is None:
        return
    if dataset_path is not None:
        path = dataset_path.expanduser().resolve()
        ok, detail = validate_dataset(path)
        (report.passed if ok else report.fail)(f"dataset/{task}", detail)
        return
    mapping = TRAIN_DATASETS if mode.startswith("train") else EVAL_DATASETS
    tasks = TASKS if mode == "train-multitask" else (task,)
    for name in tasks:
        path = cache / mapping[str(name)]
        ok, detail = validate_dataset(path)
        (report.passed if ok else report.fail)(f"dataset/{name}", detail)


def resolve_checkpoint(policy: str, cache: Path) -> Path | None:
    candidate = Path(policy).expanduser()
    if not candidate.is_absolute():
        candidate = cache / "checkpoints" / candidate
    candidate = candidate.resolve()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        weights = sorted(candidate.glob("*.pt"))
        if len(weights) == 1:
            return weights[0]
    return None


def check_checkpoint(report: Report, policy: str | None, cache: Path | None) -> None:
    if not policy:
        report.fail("checkpoint", "evaluation requires --policy")
        return
    if cache is None:
        return
    checkpoint = resolve_checkpoint(policy, cache)
    if checkpoint is None:
        report.fail("checkpoint", f"cannot resolve one .pt file from policy={policy}")
        return
    try:
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError("state dict is empty or invalid")
    except Exception as exc:
        report.fail("checkpoint", f"cannot load {checkpoint}: {exc}")
        return
    report.passed(
        "checkpoint",
        f"{checkpoint} ({len(state)} tensors, sha256={sha256(checkpoint)[:12]})",
    )
    config = checkpoint.parent / "config.yaml"
    if not config.exists():
        config = checkpoint.parent / "config.json"
    if not config.is_file():
        report.warn("checkpoint config", "config.yaml/config.json is missing")
    else:
        text = config.read_text(errors="replace")
        if "jepa.JEPA" in text and "four_slot" in text:
            report.passed("checkpoint config", f"JEPA four-slot config: {config.name}")
        else:
            report.warn("checkpoint config", f"verify JEPA/four-slot fields manually: {config}")


def check_gpus(report: Report, mode: str, strict_resources: bool, min_free_gib: float | None) -> None:
    try:
        import torch
    except ImportError:
        return
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    required = 4 if mode == "train-multitask" else 1
    if count < required:
        report.fail("GPU count", f"requires {required}, visible {count}")
        return
    report.passed("GPU count", f"requires {required}, visible {count}")
    threshold = min_free_gib if min_free_gib is not None else (80.0 if mode.startswith("train") else 1.0)
    low = []
    details = []
    for index in range(required):
        free, total = torch.cuda.mem_get_info(index)
        free_gib = free / 2**30
        total_gib = total / 2**30
        details.append(f"gpu{index}={free_gib:.1f}/{total_gib:.1f}GiB free")
        if free_gib < threshold:
            low.append(index)
    if low:
        fn = report.fail if strict_resources else report.warn
        fn("GPU memory", f"recommended >= {threshold:.1f} GiB; " + ", ".join(details))
    else:
        report.passed("GPU memory", ", ".join(details))


def check_clear(report: Report, args) -> None:
    if not args.clear_root:
        report.fail("CLEAR root", "eval-clear requires --clear-root")
        return
    root = Path(args.clear_root).expanduser().resolve()
    package = root / "clear_lewm"
    if not package.is_dir():
        report.fail("CLEAR root", f"missing clear_lewm package: {root}")
        return
    version_text = (root / "pyproject.toml").read_text(errors="replace") if (root / "pyproject.toml").exists() else ""
    if f'version = "{CLEAR_VERSION}"' not in version_text:
        report.fail("CLEAR version", f"expected {CLEAR_VERSION} in {root / 'pyproject.toml'}")
    else:
        report.passed("CLEAR version", CLEAR_VERSION)
    if (root / ".git").exists():
        commit = command(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
        if commit != CLEAR_COMMIT:
            report.fail("CLEAR commit", f"expected {CLEAR_COMMIT}, found {commit}")
        else:
            report.passed("CLEAR commit", commit)
        dirty = command(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root).stdout.strip()
        if dirty:
            report.fail("CLEAR worktree", "tracked files are modified")
        else:
            report.passed("CLEAR worktree", "tracked files are clean")
    else:
        report.warn(
            "CLEAR commit",
            f"git metadata unavailable; verified version {CLEAR_VERSION} but cannot prove commit {CLEAR_COMMIT}",
        )
    manifest = root / "manifests" / "v0.5" / args.task / f"moderate-seed{args.eval_seed}-n100.json"
    if manifest.is_file():
        report.passed("CLEAR manifest", f"{manifest.relative_to(root)} sha256={sha256(manifest)[:12]}")
    else:
        report.fail("CLEAR manifest", f"missing {manifest}")
    if not args.upstream_root:
        report.fail("upstream root", "eval-clear requires --upstream-root")
    else:
        upstream = Path(args.upstream_root).expanduser().resolve()
        if (
            (upstream / "config").is_dir()
            and (upstream / "eval.py").is_file()
            and (upstream / "jepa.py").is_file()
        ):
            report.passed("upstream root", str(upstream))
        else:
            report.fail("upstream root", f"missing LeWM config/eval.py/jepa.py in {upstream}")


def run_tests(report: Report, skip: bool) -> None:
    if skip:
        report.warn("unit tests", "skipped by --skip-tests")
        return
    result = command([sys.executable, "-m", "pytest", "-q", "tests"])
    output = (result.stdout + result.stderr).strip().splitlines()
    detail = output[-1] if output else f"exit={result.returncode}"
    if result.returncode == 0:
        report.passed("unit tests", detail)
    else:
        report.fail("unit tests", detail)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("train-single", "train-multitask", "eval-official", "eval-clear"))
    result.add_argument("--task", choices=TASKS)
    result.add_argument("--cache-dir", type=Path)
    result.add_argument("--policy", help="checkpoint path or path relative to cache/checkpoints")
    result.add_argument("--dataset-path", type=Path, help="explicit evaluation dataset path")
    result.add_argument("--clear-root", type=Path)
    result.add_argument("--upstream-root", type=Path)
    result.add_argument("--train-seed", type=int, choices=TRAIN_SEEDS, default=3072)
    result.add_argument("--eval-seed", type=int, choices=EVAL_SEEDS, default=0)
    result.add_argument("--skip-tests", action="store_true")
    result.add_argument("--strict-git-clean", action="store_true")
    result.add_argument("--strict-resources", action="store_true")
    result.add_argument("--min-free-gib", type=float)
    result.add_argument("--json-output", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.mode != "train-multitask" and not args.task:
        raise SystemExit(f"{args.mode} requires --task")
    report = Report(args.mode)
    if args.mode.startswith("train"):
        report.passed("training seed", f"{args.train_seed} in canonical {TRAIN_SEEDS}")
    else:
        report.passed("evaluation seed", f"{args.eval_seed} in canonical {EVAL_SEEDS}")
    check_git(report, args.strict_git_clean)
    check_integrity(report)
    check_python(report)
    check_math_contract(report)
    if args.mode.startswith("train"):
        check_config_contract(report, args.mode)
    cache = resolve_cache(args, report)
    check_datasets(report, args.mode, args.task, cache, args.dataset_path)
    if args.mode.startswith("eval"):
        check_checkpoint(report, args.policy, cache)
    if args.mode == "eval-clear":
        check_clear(report, args)
    check_gpus(report, args.mode, args.strict_resources, args.min_free_gib)
    run_tests(report, args.skip_tests)
    report.print()
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report.payload(), indent=2, sort_keys=True) + "\n")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
