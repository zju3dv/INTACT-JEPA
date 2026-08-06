#!/usr/bin/env python3
"""Build and verify high-resolution editions of the INTACT project films."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2


PROJECT_URL = "https://zju3dv.github.io/INTACT-JEPA/"
FPS = 30
FORMATS = {
    "2k": (2560, 1440, "16", "5.0", "0.25"),
    "4k": (3840, 2160, "18", "5.1", "0.18"),
}
EDITIONS = {
    "page": ("intact-hero-film.mp4", "intact-hero-film", False),
    "share": ("intact-project-film-share.mp4", "intact-project-film-share", True),
}


def build(
    source: Path,
    destination: Path,
    width: int,
    height: int,
    crf: str,
    level: str,
    sharpen: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-an",
            "-vf",
            f"scale={width}:{height}:flags=lanczos,unsharp=5:5:{sharpen}:3:3:0.0,setsar=1,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            crf,
            "-profile:v",
            "high",
            "-level:v",
            level,
            "-g",
            str(FPS * 2),
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


def verify(
    path: Path,
    expected_width: int,
    expected_height: int,
    expect_qr: bool,
) -> None:
    capture = cv2.VideoCapture(str(path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    final_frame = None
    ok = True
    if expect_qr:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - fps))
        ok, final_frame = capture.read()
    capture.release()
    if (width, height, fps) != (expected_width, expected_height, FPS):
        raise RuntimeError(f"Unexpected output format: {width}x{height} at {fps} fps")
    if expect_qr and not ok:
        raise RuntimeError("Could not decode the verification frame")
    if expect_qr:
        decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(final_frame)
        if decoded != PROJECT_URL:
            raise RuntimeError(f"Encoded QR code did not validate: {decoded!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resolution", choices=("2k", "4k", "both"), default="2k")
    parser.add_argument("--edition", choices=("page", "share", "both"), default="share")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    assets = repo / "docs/assets"
    formats = FORMATS if args.resolution == "both" else {args.resolution: FORMATS[args.resolution]}
    editions = EDITIONS if args.edition == "both" else {args.edition: EDITIONS[args.edition]}
    for edition, (source_name, output_stem, expect_qr) in editions.items():
        source = assets / source_name
        for label, (width, height, crf, level, sharpen) in formats.items():
            destination = assets / f"{output_stem}-{label}.mp4"
            build(source, destination, width, height, crf, level, sharpen)
            verify(destination, width, height, expect_qr)
            suffix = " with QR" if expect_qr else ""
            print(
                f"Built and verified {edition} edition {destination} "
                f"at {width}x{height}{suffix}"
            )


if __name__ == "__main__":
    main()
