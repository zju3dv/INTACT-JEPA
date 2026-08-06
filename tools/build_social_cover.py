#!/usr/bin/env python3
"""Render native 4K and 2K social covers for INTACT."""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


WIDTH = 3840
HEIGHT = 2160

NIGHT = "#050914"
WHITE = "#FFFFFF"
PALE = "#D9E8F7"
MUTED = "#AFC0D5"
BLUE = "#279EF0"
BLUE_BRIGHT = "#58A8EA"
CORAL = "#F0525D"
INK_LINE = "#20324A"

FONT_REGULAR = "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf"
FONT_MEDIUM = "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Medium.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf"
FONT_CJK_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_CJK_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


@lru_cache(maxsize=None)
def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    path = {
        "regular": FONT_REGULAR,
        "medium": FONT_MEDIUM,
        "bold": FONT_BOLD,
    }[weight]
    return ImageFont.truetype(path, size=size)


@lru_cache(maxsize=None)
def cjk_font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    path = FONT_CJK_BOLD if weight == "bold" else FONT_CJK_REGULAR
    return ImageFont.truetype(path, size=size, index=2)


def rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = image.convert("RGBA")
    bounds = image.getchannel("A").getbbox()
    if bounds:
        image = image.crop(bounds)
    image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return image


def monochrome(image: Image.Image, remove_white: bool = False) -> Image.Image:
    image = image.convert("RGBA")
    pixels = np.asarray(image).copy()
    if remove_white:
        source = Image.fromarray(pixels[:, :, :3])
        difference = ImageChops.difference(source, Image.new("RGB", source.size, "white")).convert("L")
        alpha = difference.point(lambda value: 0 if value < 4 else min(255, int((value - 4) * 3.5)))
        pixels[:, :, 3] = np.minimum(pixels[:, :, 3], np.asarray(alpha))
    pixels[:, :, :3] = rgb("#EAF1FA")
    return fit(Image.fromarray(pixels), image.width, image.height)


def mix_color(left: tuple[int, int, int], right: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(round(a + (b - a) * amount) for a, b in zip(left, right))


def draw_particle_field(
    canvas: Image.Image,
    center: tuple[float, float],
    scale: tuple[float, float],
    count: int = 2600,
) -> None:
    width, height = canvas.size
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    rotation_y = -0.42
    rotation_x = -0.18
    cy, sy = math.cos(rotation_y), math.sin(rotation_y)
    cx, sx = math.cos(rotation_x), math.sin(rotation_x)
    projected = []

    for index in range(count):
        ratio = (index + 0.5) / count
        y = 1.0 - 2.0 * ratio
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        angle = index * golden_angle
        shell = 1.02 + 0.16 * math.sin(index * 12.9898) + 0.05 * math.sin(index * 0.31)
        x = math.cos(angle) * radial * shell
        z = math.sin(angle) * radial * shell
        y *= shell

        x, z = x * cy + z * sy, -x * sy + z * cy
        y, z = y * cx - z * sx, y * sx + z * cx
        perspective = 1.0 / max(1.75 - 0.24 * z, 0.9)
        screen_x = center[0] + x * perspective * scale[0]
        screen_y = center[1] - y * perspective * scale[1]
        color_mix = 0.5 + 0.5 * math.sin(angle * 0.72 + y * 3.4)
        color = rgb(PALE) if index % 31 == 0 else mix_color(rgb(BLUE), rgb(CORAL), color_mix)
        radius = (4.2 + 3.6 * (z + 1.35) / 2.7) * (1.0 + 0.12 * math.sin(index * 0.31))
        alpha = round(105 + 110 * max(0.0, min(1.0, (z + 1.35) / 2.7)))
        projected.append((z, screen_x, screen_y, radius, color, alpha))

    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    points = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    point_draw = ImageDraw.Draw(points)
    for z, x, y, radius, color, alpha in sorted(projected):
        if x < -30 or x > width + 30 or y < -30 or y > height + 30:
            continue
        if alpha > 185 and int((x + y)) % 7 == 0:
            glow_radius = radius * 3.2
            glow_draw.ellipse(
                (x - glow_radius, y - glow_radius, x + glow_radius, y + glow_radius),
                fill=(*color, 52),
            )
        point_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, alpha))

    glow = glow.filter(ImageFilter.GaussianBlur(18))
    canvas.alpha_composite(glow)
    canvas.alpha_composite(points)


def draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    size: int,
    color: str,
    weight: str = "regular",
    anchor: str = "mm",
) -> None:
    draw.text(position, value, font=font(size, weight), fill=color, anchor=anchor)


def draw_cjk_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    size: int,
    color: str,
    weight: str = "regular",
    anchor: str = "mm",
) -> None:
    draw.text(position, value, font=cjk_font(size, weight), fill=color, anchor=anchor)


def draw_centered_runs(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    runs: tuple[tuple[str, str], ...],
    size: int,
) -> None:
    selected_font = cjk_font(size, "bold")
    widths = [draw.textlength(value, font=selected_font) for value, _ in runs]
    x = center[0] - sum(widths) / 2
    for width, (value, color) in zip(widths, runs):
        draw.text((x, center[1]), value, font=selected_font, fill=color, anchor="lm")
        x += width


def draw_affiliations(canvas: Image.Image, assets: Path) -> None:
    marks = (
        (
            "ZHEJIANG UNIVERSITY",
            monochrome(Image.open(assets / "zhejiang-university-logo.png"), remove_white=True),
        ),
        (
            "TSINGHUA UNIVERSITY",
            monochrome(Image.open(assets / "tsinghua-university-logo.jpg"), remove_white=True),
        ),
        ("INSPATIO", monochrome(Image.open(assets / "inspatio-logo.png"))),
        ("ROBOPARTY LAB", monochrome(Image.open(assets / "roboparty-lab-logo.png"))),
    )
    centers = (510, 1450, 2390, 3330)
    for center_x, (name, mark) in zip(centers, marks):
        max_width = 410 if name in {"INSPATIO", "ROBOPARTY LAB"} else 150
        max_height = 126 if name in {"INSPATIO", "ROBOPARTY LAB"} else 132
        mark = fit(mark, max_width, max_height)
        canvas.alpha_composite(mark, (center_x - mark.width // 2, 1922 - mark.height // 2))
        draw_text(ImageDraw.Draw(canvas), (center_x, 2030), name, 25, MUTED, "medium")


def render_cover(repo: Path) -> Image.Image:
    assets = repo / "docs/assets"
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), (*rgb(NIGHT), 255))
    draw_particle_field(canvas, center=(1920, 870), scale=(1370, 1060))

    veil = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(veil).ellipse((620, 240, 3220, 1500), fill=(*rgb(NIGHT), 146))
    canvas.alpha_composite(veil)

    draw = ImageDraw.Draw(canvas)
    draw.line((760, 224, 1320, 224), fill=BLUE_BRIGHT, width=5)
    draw.line((2520, 224, 3080, 224), fill=CORAL, width=5)
    draw_text(draw, (1920, 224), "WORLD MODELS · LATENT INTENT · DIRECT CONTROL", 42, "#F7FAFF", "medium")

    logo = fit(Image.open(assets / "intact-wordmark-light.png"), 1780, 610)
    canvas.alpha_composite(logo, (1920 - logo.width // 2, 690 - logo.height // 2))

    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (1920, 1180), "ISOMORPHIC INTENT-TO-ACTION LEARNING", 84, WHITE, "bold")
    draw.line((1290, 1258, 2550, 1258), fill=BLUE_BRIGHT, width=5)
    draw_text(draw, (1920, 1344), "Search-Free Control from a Learned World Model", 55, "#EEF4FC")

    metric_centers = (550, 1440, 2350, 3290)
    metrics = (
        ("1", "FULL-DATA EPOCH", WHITE),
        ("95.33%", "DIRECT MACRO SR", CORAL),
        ("0", "SEARCH", WHITE),
        ("2.9–5.5 ms", "INFERENCE", BLUE_BRIGHT),
    )
    draw.line((240, 1484, 3600, 1484), fill=INK_LINE, width=3)
    draw.line((240, 1746, 3600, 1746), fill=INK_LINE, width=3)
    for index, (center_x, (value, label, color)) in enumerate(zip(metric_centers, metrics)):
        if index:
            divider_x = (metric_centers[index - 1] + center_x) // 2
            draw.line((divider_x, 1535, divider_x, 1695), fill=INK_LINE, width=3)
        draw_text(draw, (center_x, 1585), value, 71, color, "bold")
        draw_text(draw, (center_x, 1668), label, 27, MUTED, "medium")

    draw_text(draw, (1920, 1812), "Junhan Sun · Hao Zhao · Guofeng Zhang", 32, "#EAF1FA", "medium")
    draw_affiliations(canvas, assets)
    draw_text(ImageDraw.Draw(canvas), (1920, 2110), "zju3dv.github.io/INTACT-JEPA", 24, "#8396B1", "medium")
    return canvas.convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    repo = parse_args().repo.resolve()
    assets = repo / "docs/assets"
    cover = render_cover(repo)
    cover.save(assets / "intact-social-cover-4k.png", optimize=True)
    cover.resize((2560, 1440), Image.Resampling.LANCZOS).save(
        assets / "intact-social-cover-2k.jpg",
        quality=96,
        subsampling=0,
        optimize=True,
    )
    print("Rendered landscape 4K PNG and 2K JPEG covers")


if __name__ == "__main__":
    main()
