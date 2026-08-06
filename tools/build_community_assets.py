#!/usr/bin/env python3
"""Build the permanent QR code for the INTACT Community landing page."""

from __future__ import annotations

from pathlib import Path

import cv2


COMMUNITY_URL = "https://zju3dv.github.io/INTACT-JEPA/community/"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "docs/assets/intact-community-page-qr.png"

    encoder = cv2.QRCodeEncoder_create()
    modules = encoder.encode(COMMUNITY_URL)
    if modules is None or not modules.size:
        raise RuntimeError("OpenCV did not generate a QR matrix")

    quiet_zone = 4
    modules = cv2.copyMakeBorder(
        modules,
        quiet_zone,
        quiet_zone,
        quiet_zone,
        quiet_zone,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    scale = 24
    image = cv2.resize(
        modules,
        (modules.shape[1] * scale, modules.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image)

    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    if decoded != COMMUNITY_URL:
        raise RuntimeError(f"Generated QR code did not validate: {decoded!r}")
    print(f"Built and verified {destination} -> {decoded}")


if __name__ == "__main__":
    main()
