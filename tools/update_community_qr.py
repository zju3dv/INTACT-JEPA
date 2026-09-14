#!/usr/bin/env python3
"""Publish a current WeChat group invitation without changing its public URL."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Current WeChat group QR screenshot")
    parser.add_argument("--valid-until", required=True, help="Expiry date in YYYY-MM-DD")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        expiry = date.fromisoformat(args.valid_until)
    except ValueError as exc:
        raise SystemExit("--valid-until must use YYYY-MM-DD") from exc
    if expiry < date.today():
        raise SystemExit("The supplied invitation is already expired")

    source = args.image.expanduser().resolve()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not decode image: {source}")

    repo = args.repo.resolve()
    assets = repo / "docs/assets"
    destination = assets / "community-wechat-current.png"
    metadata = assets / "community-invite.json"
    assets.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise SystemExit(f"Could not write {destination}")

    payload = {
        "status": "active",
        "updated": date.today().isoformat(),
        "validUntil": expiry.isoformat(),
        "image": "../assets/community-wechat-current.png",
        "contactEmail": "luoliibaqi4747@gmail.com",
    }
    metadata.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    result = "QR payload decoded" if decoded else "image published; verify in WeChat"
    print(f"Updated {destination} through {expiry.isoformat()} ({result})")


if __name__ == "__main__":
    main()
