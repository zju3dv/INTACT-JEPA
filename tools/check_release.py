#!/usr/bin/env python3
"""Run lightweight release-hygiene checks without third-party dependencies."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", "__pycache__"}
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".h5", ".hdf5", ".pem", ".pt", ".pth"}
MAX_FILE_BYTES = 20 * 1024 * 1024
PRIVATE_COLLAB_FILES = {
    "TEAM_PROGRESS.md",
    "docs/COLLABORATION.md",
    "docs/FLEET.md",
}


def files() -> list[Path]:
    allow_private_collab = os.environ.get("INTACT_RELEASE_ALLOW_PRIVATE_COLLAB") == "1"
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not IGNORED_PARTS.intersection(path.parts)
        and not (
            allow_private_collab
            and path.relative_to(ROOT).as_posix() in PRIVATE_COLLAB_FILES
        )
    )


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {
        ".gitattributes",
        ".gitignore",
        "LICENSE",
    }


def local_targets(text: str) -> list[str]:
    markdown = re.findall(r"!?(?:\[[^\]]*\])\(([^)]+)\)", text)
    html = re.findall(r"(?:href|src)=[\"']([^\"']+)[\"']", text)
    return markdown + html


def main() -> int:
    errors: list[str] = []
    required = {
        "README.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "LICENSE",
        "NOTICE.md",
        "docs/METHOD.md",
        "docs/REPRODUCIBILITY.md",
        "docs/RESULTS.md",
    }
    present = {path.relative_to(ROOT).as_posix() for path in files()}
    for missing in sorted(required - present):
        errors.append(f"missing required file: {missing}")

    absolute_path = re.compile(
        r"(?<![A-Za-z0-9_.-])/(?:data|mnt|home)/[A-Za-z0-9_.-]+"
    )
    host_prefixes = ("rp" + "pro", "h2" + "0", "h20" + "0")
    infrastructure_host = re.compile(
        rf"\b(?:{'|'.join(host_prefixes)})[-_]?\d+\b", re.I
    )
    private_key = re.compile(r"BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY")
    github_token = re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b")

    for path in files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 20 MiB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden release artifact: {relative}")
        if not is_text(path):
            continue

        text = path.read_text(encoding="utf-8")
        for label, pattern in (
            ("absolute machine path", absolute_path),
            ("internal infrastructure hostname", infrastructure_host),
            ("private key", private_key),
            ("GitHub token", github_token),
        ):
            if pattern.search(text):
                errors.append(f"{label} in {relative}")

        if path.suffix.lower() != ".md":
            continue
        for target in local_targets(text):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            candidate = (path.parent / target).resolve()
            if not candidate.exists():
                errors.append(f"broken local link in {relative}: {target}")

    if errors:
        print("Release checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Release checks passed for {len(present)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
