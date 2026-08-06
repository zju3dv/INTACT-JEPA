#!/usr/bin/env python3
"""Build the cinematic INTACT website film from audited local artifacts.

The public repository stores the rendered film, but not model checkpoints or
raw research data. This builder reads paired rollout videos and the exported
latent projection used by the website, then renders every title, plot, and
diagram natively at 1080p.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 30

NIGHT = "#07111F"
NIGHT_SOFT = "#0E1A2B"
PAPER = "#F4F7FB"
WHITE = "#FFFFFF"
INK = "#111B2E"
MUTED = "#65738A"
LINE = "#D8E0EA"
BLUE = "#367FBE"
BLUE_BRIGHT = "#58A8EA"
BLUE_PALE = "#DDECF8"
CORAL = "#EA5A60"
CORAL_PALE = "#FBE1E3"
FILM_RED = "#FF4654"
TEAL = "#17A88E"
PURPLE = "#7152D9"
GOLD = "#E7A33B"
LOCAL_ORANGE = "#E8762D"
LOCAL_PALE = "#FCE4D2"
GOAL_GREEN = "#168675"
GOAL_PALE = "#DDF2EC"

TASKS = ("PushT", "Cube", "Reacher", "TwoRoom")
TASK_KEYS = ("pusht", "cube", "reacher", "tworoom")
TASK_COLORS = (PURPLE, TEAL, GOLD, CORAL)
EPISODES = {"pusht": 99, "cube": 36, "reacher": 78, "tworoom": 66}

LEWM_SR = (96.0, 74.0, 86.0, 87.0)
INTACT_SR = (85.78, 100.0, 97.67, 97.89)
GUARDED_SR = (92.22, 99.78, 97.44, 98.0)
LEWM_MACRO = 85.75
INTACT_MACRO = 95.33
GUARDED_MACRO = 96.86

PROJECT_URL = "https://zju3dv.github.io/INTACT-JEPA/"
GITHUB_URL = "https://github.com/zju3dv/INTACT-JEPA"
INSPATIO_URL = "https://www.inspatio.com/"
ROBOPARTY_URL = "https://lab.roboparty.com/"

FONT_REGULAR = "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf"
FONT_MEDIUM = "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Medium.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_MATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@lru_cache(maxsize=None)
def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    path = {
        "regular": FONT_REGULAR,
        "medium": FONT_MEDIUM,
        "bold": FONT_BOLD,
        "mono": FONT_MONO,
        "math": FONT_MATH,
    }[weight]
    return ImageFont.truetype(path, size=size)


def rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smooth(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * clamp(amount)


def reveal(t: float, start: float, duration: float = 0.65) -> float:
    return smooth((t - start) / duration)


def base(color: str) -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), color)


def text(
    draw: ImageDraw.ImageDraw,
    xy,
    value: str,
    size: int,
    color: str,
    weight: str = "regular",
    anchor: str | None = None,
) -> None:
    draw.text(xy, value, font=font(size, weight), fill=color, anchor=anchor)


def colored_text_runs(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    runs: tuple[tuple[str, str], ...],
    size: int,
    weight: str = "regular",
) -> float:
    x, y = xy
    fnt = font(size, weight)
    for value, color in runs:
        draw.text((x, y), value, font=fnt, fill=color)
        x += draw.textlength(value, font=fnt)
    return x


def alpha_layer(frame: Image.Image, painter, opacity: float = 1.0) -> Image.Image:
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    painter(ImageDraw.Draw(overlay), overlay)
    if opacity < 1.0:
        overlay.putalpha(overlay.getchannel("A").point(lambda value: int(value * clamp(opacity))))
    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")


def blend(left: Image.Image, right: Image.Image, amount: float) -> Image.Image:
    return Image.blend(left, right, smooth(amount))


def fit_rgba(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    return fitted


def film_logo(image: Image.Image) -> Image.Image:
    pixels = np.asarray(image.convert("RGBA")).copy()
    red = pixels[:, :, 0].astype(np.int16)
    green = pixels[:, :, 1].astype(np.int16)
    blue = pixels[:, :, 2].astype(np.int16)
    red_mask = (red > 145) & (red > green * 1.22) & (red > blue * 1.15) & (pixels[:, :, 3] > 0)
    pixels[red_mask, :3] = rgb(FILM_RED)
    return Image.fromarray(pixels)


def draw_film_overline(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    outer_half_width: int,
    text_half_width: int,
    size: int,
) -> None:
    draw.line((center_x - outer_half_width, y, center_x - text_half_width, y), fill=BLUE_BRIGHT, width=3)
    draw.line((center_x + text_half_width, y, center_x + outer_half_width, y), fill=FILM_RED, width=3)
    text(draw, (center_x, y), "World models · latent intent · direct control", size, "#F7FAFF", "medium", "mm")


def paste_cover(canvas: Image.Image, source: Image.Image, box) -> None:
    x0, y0, x1, y1 = map(int, box)
    target_w, target_h = x1 - x0, y1 - y0
    ratio = max(target_w / source.width, target_h / source.height)
    resized = source.resize(
        (math.ceil(source.width * ratio), math.ceil(source.height * ratio)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    canvas.paste(resized.crop((left, top, left + target_w, top + target_h)), (x0, y0))


def rounded(draw: ImageDraw.ImageDraw, box, radius: int, fill: str, outline: str | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def rounded_horizontal_gradient(frame: Image.Image, box, radius: int, left: str, right: str) -> None:
    x0, y0, x1, y1 = map(int, box)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    start = np.asarray(rgb(left), dtype=np.float32)
    end = np.asarray(rgb(right), dtype=np.float32)
    row = np.linspace(start, end, width).astype(np.uint8)
    pixels = np.repeat(row[np.newaxis, :, :], height, axis=0)
    panel = Image.fromarray(pixels)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    frame.paste(panel, (x0, y0), mask)


def arrow(draw: ImageDraw.ImageDraw, start, end, color: str, width: int = 7, head: int = 22) -> None:
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    left = (x1 - head * math.cos(angle - 0.55), y1 - head * math.sin(angle - 0.55))
    right = (x1 - head * math.cos(angle + 0.55), y1 - head * math.sin(angle + 0.55))
    draw.polygon((end, left, right), fill=color)


def curved_arrow(
    draw: ImageDraw.ImageDraw,
    start,
    control,
    end,
    color: str,
    width: int = 7,
    head: int = 22,
) -> None:
    points = []
    for value in np.linspace(0.0, 1.0, 48):
        inverse = 1.0 - value
        x = inverse * inverse * start[0] + 2 * inverse * value * control[0] + value * value * end[0]
        y = inverse * inverse * start[1] + 2 * inverse * value * control[1] + value * value * end[1]
        points.append((x, y))
    draw.line(points, fill=color, width=width, joint="curve")
    tangent = (end[0] - control[0], end[1] - control[1])
    angle = math.atan2(tangent[1], tangent[0])
    left = (end[0] - head * math.cos(angle - 0.55), end[1] - head * math.sin(angle - 0.55))
    right = (end[0] - head * math.cos(angle + 0.55), end[1] - head * math.sin(angle + 0.55))
    draw.polygon((end, left, right), fill=color)


def dashed_ellipse(draw: ImageDraw.ImageDraw, box, color: str, width: int, phase: float) -> None:
    for start in range(0, 360, 24):
        offset = int(phase * 22) % 24
        draw.arc(box, start=start + offset, end=start + offset + 13, fill=color, width=width)


def dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box,
    color: str,
    width: int = 4,
    dash: int = 18,
    gap: int = 12,
) -> None:
    x0, y0, x1, y1 = map(int, box)
    for start in range(x0, x1, dash + gap):
        draw.line((start, y0, min(start + dash, x1), y0), fill=color, width=width)
        draw.line((start, y1, min(start + dash, x1), y1), fill=color, width=width)
    for start in range(y0, y1, dash + gap):
        draw.line((x0, start, x0, min(start + dash, y1)), fill=color, width=width)
        draw.line((x1, start, x1, min(start + dash, y1)), fill=color, width=width)


def draw_split_name(draw: ImageDraw.ImageDraw, center_x: float, y: float, size: int) -> None:
    parts = (("IN", BLUE), ("T", INK), ("ACT", CORAL))
    fnt = font(size, "bold")
    widths = [draw.textlength(value, font=fnt) for value, _ in parts]
    x = center_x - sum(widths) / 2
    for (value, color), width in zip(parts, widths):
        draw.text((x, y), value, font=fnt, fill=color, anchor="lm")
        x += width


def crop_agent(frame: Image.Image) -> Image.Image:
    return frame.crop((16, 16, 240, 240))


def crop_goal(frame: Image.Image) -> Image.Image:
    return frame.crop((492, 16, 716, 240))


def crop_reference(frame: Image.Image) -> Image.Image:
    return frame.crop((254, 16, 478, 240))


def read_video(path: Path) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    frames: list[Image.Image] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


@dataclass
class RolloutPair:
    task: str
    key: str
    episode: int
    lewm: list[Image.Image]
    intact: list[Image.Image]

    def at(self, progress: float) -> tuple[Image.Image, Image.Image, Image.Image]:
        index = min(len(self.lewm) - 1, int(clamp(progress) * (len(self.lewm) - 1)))
        intact_index = min(len(self.intact) - 1, int(clamp(progress) * (len(self.intact) - 1)))
        shared_start = crop_reference(self.intact[0])
        rollout_mix = smooth(progress / 0.14)
        lewm = Image.blend(shared_start, crop_agent(self.lewm[index]), rollout_mix)
        intact = Image.blend(shared_start, crop_agent(self.intact[intact_index]), rollout_mix)
        return lewm, intact, crop_goal(self.intact[0])


def load_particle_cloud(root: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata = json.loads((root / "complete/metadata.json").read_text())
    shape = (metadata["frames"], metadata["points"], 3)
    coords = np.memmap(root / "complete/coordinates-3d.i16", dtype="<i2", mode="r", shape=shape)
    points = np.asarray(coords[-1], dtype=np.float32).copy()
    points -= np.mean(points, axis=0, keepdims=True)
    scale = max(float(np.quantile(np.linalg.norm(points, axis=1), 0.98)), 1.0)
    points /= scale
    labels = np.repeat(np.arange(4), math.ceil(len(points) / 4))[: len(points)]
    return points, labels


def load_latent_movie(root: Path) -> tuple[np.memmap, np.ndarray, np.ndarray]:
    metadata = json.loads((root / "complete/metadata.json").read_text())
    shape = (metadata["frames"], metadata["points"], 2)
    coordinates = np.memmap(root / "complete/coordinates-2d.i16", dtype="<i2", mode="r", shape=shape)
    epochs = np.asarray(metadata["epochs"], dtype=np.float32)
    effective_rank = np.asarray(metadata["effective_rank"], dtype=np.float32)
    return coordinates, epochs, effective_rank


def project(points: np.ndarray, angle: float, tilt: float = -0.36) -> tuple[np.ndarray, np.ndarray]:
    ca, sa = math.cos(angle), math.sin(angle)
    cb, sb = math.cos(tilt), math.sin(tilt)
    ry = np.array(((ca, 0, sa), (0, 1, 0), (-sa, 0, ca)), dtype=np.float32)
    rx = np.array(((1, 0, 0), (0, cb, -sb), (0, sb, cb)), dtype=np.float32)
    rotated = points @ ry.T @ rx.T
    depth = rotated[:, 2]
    perspective = 1.0 / np.maximum(1.55 - 0.22 * depth, 0.85)
    return rotated[:, :2] * perspective[:, None], depth


def draw_particle_space(
    frame: Image.Image,
    points: np.ndarray,
    labels: np.ndarray,
    t: float,
    center=(960, 520),
    scale: float = 510,
    opacity: float = 1.0,
) -> Image.Image:
    projected, depth = project(points, 0.35 + t * 0.17)

    def paint(draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
        order = np.argsort(depth)
        for rank, index in enumerate(order):
            x = center[0] + projected[index, 0] * scale
            y = center[1] - projected[index, 1] * scale
            pulse = 1.0 + 0.16 * math.sin(t * 1.9 + index * 0.31)
            radius = (3.1 + 2.0 * (rank / max(1, len(order) - 1))) * pulse
            color = rgb(TASK_COLORS[int(labels[index])])
            alpha = int(75 + 120 * (rank / max(1, len(order) - 1)))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, alpha))
            if index % 29 == 0:
                tail_x = center[0] + project(points[index : index + 1], 0.35 + (t - 0.12) * 0.17)[0][0, 0] * scale
                tail_y = center[1] - project(points[index : index + 1], 0.35 + (t - 0.12) * 0.17)[0][0, 1] * scale
                draw.line((tail_x, tail_y, x, y), fill=(*color, 75), width=2)

    return alpha_layer(frame, paint, opacity)


def scene_opening(
    t: float,
    points: np.ndarray,
    labels: np.ndarray,
    logo: Image.Image,
    institution_marks: tuple[tuple[str, Image.Image], ...] = (),
) -> Image.Image:
    frame = base(NIGHT)
    frame = draw_particle_space(frame, points, labels, t, center=(960, 485), scale=625, opacity=0.84)

    def veil(draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
        draw.ellipse((380, 180, 1540, 860), fill=(*rgb(NIGHT), 138))

    frame = alpha_layer(frame, veil, reveal(t, 0.2, 1.2))
    fade = 0.84 + 0.16 * reveal(t, 0.2, 0.9)

    def title_layer(draw: ImageDraw.ImageDraw, overlay: Image.Image) -> None:
        draw_film_overline(draw, 960, 268, 570, 270, 23)
        mark = fit_rgba(logo, 870, 300)
        overlay.alpha_composite(mark, (960 - mark.width // 2, 510 - mark.height // 2))
        text(draw, (960, 710), "ISOMORPHIC INTENT-TO-ACTION LEARNING", 32, WHITE, "medium", "ma")
        draw.line((670, 756, 1250, 756), fill=(*rgb(BLUE_BRIGHT), 180), width=3)
        text(draw, (960, 802), "Search-Free Control from a Learned World Model", 27, "#EEF4FC", "regular", "ma")

    frame = alpha_layer(frame, title_layer, fade)
    if institution_marks:
        marks_frame = draw_institution_row(
            frame,
            institution_marks,
            centers=(435, 785, 1135, 1485),
            mark_y=930,
            name_y=990,
            max_width=190,
            max_height=62,
            name_size=13,
        )
        frame = blend(frame, marks_frame, reveal(t, 1.05, 0.9))
    return frame


def scene_demos(t: float, pairs: list[RolloutPair]) -> Image.Image:
    frame = base(NIGHT)
    draw = ImageDraw.Draw(frame)
    colored_text_runs(
        draw,
        (84, 62),
        (
            ("SAME START. SAME GOAL. ", WHITE),
            ("IN", BLUE_BRIGHT),
            ("T", WHITE),
            ("ACT", FILM_RED),
            (" WIN!", WHITE),
        ),
        45,
        "bold",
    )
    rounded(draw, (1088, 42, 1450, 112), 8, "#10273D", "#295B86", 2)
    text(draw, (1112, 60), "PLANNER LATENCY", 14, "#8FBCE2", "bold")
    text(draw, (1112, 80), "≈300× lower · 2.9–5.5 ms", 22, BLUE_BRIGHT, "bold")
    rounded(draw, (1472, 42, 1836, 112), 8, "#2A1721", "#7A313D", 2)
    text(draw, (1496, 60), "STABLE DIRECT CONTROL", 14, "#F3A3A8", "bold")
    text(draw, (1496, 80), "95.33 ± 0.58% macro SR", 22, CORAL, "bold")
    draw.line((84, 137, 1836, 137), fill="#20324A", width=2)
    progress = smooth((t - 0.75) / 7.7)
    locations = ((72, 180), (996, 180), (72, 605), (996, 605))

    for pair, (x, y), accent in zip(pairs, locations, TASK_COLORS):
        lewm, intact, goal = pair.at(progress)
        group_box = (x, y, x + 852, y + 380)
        rounded(draw, group_box, 16, NIGHT_SOFT, "#21334B", 2)
        text(draw, (x + 26, y + 42), pair.task.upper(), 24, accent, "bold")
        text(draw, (x + 26, y + 77), f"episode {pair.episode:02d}", 15, "#7F91AA", "mono")
        video_y = y + 25
        lewm_box = (x + 150, video_y, x + 480, video_y + 330)
        intact_box = (x + 500, video_y, x + 830, video_y + 330)
        paste_cover(frame, lewm, lewm_box)
        paste_cover(frame, intact, intact_box)
        draw = ImageDraw.Draw(frame)
        draw.rectangle(lewm_box, outline="#6F7C90", width=3)
        draw.rectangle(intact_box, outline=CORAL, width=4)

        def label_bar(box, label, color):
            bx0, by0, bx1, _ = box
            draw.rectangle((bx0, by0, bx1, by0 + 44), fill=NIGHT)
            text(draw, ((bx0 + bx1) / 2, by0 + 23), label, 17, color, "bold", "mm")

        label_bar(lewm_box, "LeWM + CEM", "#D2D9E4")
        label_bar(intact_box, "INTACT · DIRECT", CORAL)
        goal_box = (x + 42, y + 246, x + 132, y + 336)
        paste_cover(frame, goal, goal_box)
        draw = ImageDraw.Draw(frame)
        draw.rectangle(goal_box, outline=WHITE, width=3)
        text(draw, (x + 87, y + 356), "GOAL", 13, "#9AAAC0", "bold", "ma")

    text(draw, (960, 1033), "Matched evaluation seed · fixed episode identity · goal held constant", 17, "#8798AF", "regular", "mm")
    return frame


def latent_points(t: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    centers = np.array(((-0.48, -0.14), (0.20, 0.36), (0.47, -0.22)))
    clouds = []
    groups = []
    for group, center in enumerate(centers):
        local = rng.normal(0.0, (0.19, 0.13), size=(58, 2)) + center
        local[:, 0] += 0.035 * np.sin(t * 1.7 + np.arange(58) * 0.27 + group)
        local[:, 1] += 0.025 * np.cos(t * 1.4 + np.arange(58) * 0.19)
        clouds.append(local)
        groups.extend([group] * len(local))
    return np.concatenate(clouds), np.asarray(groups)


def spherical_latent_points(t: float, count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float32)
    y = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    angle = indices * (math.pi * (3.0 - math.sqrt(5.0))) + t * 0.22
    points = np.column_stack((np.cos(angle) * radius, y, np.sin(angle) * radius))
    ca, sa = math.cos(t * 0.14), math.sin(t * 0.14)
    rotated_x = ca * points[:, 0] + sa * points[:, 2]
    depth = -sa * points[:, 0] + ca * points[:, 2]
    perspective = 1.0 / np.maximum(1.4 - depth * 0.2, 0.9)
    return np.column_stack((rotated_x * perspective, points[:, 1] * perspective)) * 0.68


def action_points(t: float) -> np.ndarray:
    rng = np.random.default_rng(29)
    angles = np.linspace(0, 2 * math.pi, 46, endpoint=False)
    radius = 0.52 + rng.normal(0, 0.065, len(angles))
    x = radius * np.cos(angles + t * 0.12)
    y = 0.55 * radius * np.sin(angles + t * 0.12)
    return np.column_stack((x, y))


def scene_method(t: float, input_frame: Image.Image) -> Image.Image:
    frame = base(PAPER)
    draw = ImageDraw.Draw(frame)
    text(draw, (86, 67), "FROM LATENT INTENT TO ACTION", 44, INK, "bold")
    draw.line((86, 137, 1834, 137), fill=LINE, width=2)

    input_alpha = reveal(t, 0.2)
    input_x = 66 + 34 * smooth((t - 1.1) / 1.5)
    input_box = (input_x, 355, input_x + 265, 620)
    if input_alpha > 0:
        paste_cover(frame, input_frame, input_box)
        draw = ImageDraw.Draw(frame)
        draw.rectangle(input_box, outline=INK, width=5)
        text(draw, (input_x + 132, 665), "RGB OBSERVATION", 19, MUTED, "bold", "ma")

    encoder_alpha = reveal(t, 0.7)
    if encoder_alpha > 0:
        rounded(draw, (402, 365, 612, 610), 18, WHITE, INK, 4)
        text(draw, (507, 466), "ENCODER", 27, INK, "bold", "mm")
        text(draw, (507, 518), "fθ", 42, BLUE, "bold", "mm")
        if t > 1.2:
            arrow(draw, (input_x + 277, 487), (394, 487), INK, 6, 16)

    latent_alpha = reveal(t, 2.0, 0.9)
    latent, groups = latent_points(t)
    morph = smooth((t - 6.6) / 1.25)
    sphere = spherical_latent_points(t, len(latent))
    latent = latent * (1.0 - morph) + sphere * morph
    latent_center = (930, 500)
    if latent_alpha > 0:
        text(draw, (930, 235), "LATENT INTENT SPACE", 22, BLUE, "bold", "ma")
        if t > 2.15:
            arrow(draw, (630, 487), (690, 487), INK, 7, 22)
        for index, point in enumerate(latent):
            x = latent_center[0] + point[0] * 310
            y = latent_center[1] - point[1] * 310
            radius = 5 + 1.3 * math.sin(index * 0.4 + t)
            color = BLUE_BRIGHT if groups[index] == 1 else BLUE
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    intent_alpha = reveal(t, 3.1, 0.8)
    ring_fade = 1.0 - smooth((t - 6.45) / 1.0)
    if intent_alpha > 0 and ring_fade > 0:
        def paint_intent_ring(ring_draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
            dashed_ellipse(ring_draw, (840, 350, 1065, 570), BLUE, 5, t)
            text(ring_draw, (952, 312), "INTENT FAMILY", 18, BLUE, "bold", "ma")

        frame = alpha_layer(frame, paint_intent_ring, ring_fade)
        draw = ImageDraw.Draw(frame)

    action_alpha = reveal(t, 4.15, 0.8)
    action_center = (1600, 500)
    if action_alpha > 0:
        curved_arrow(draw, (1170, 470), (1300, 335), (1432, 470), INK, 8, 25)
        text(draw, (1280, 352), "πθ", 28, INK, "bold", "mm")
        text(draw, (1600, 235), "ACTION SPACE", 22, CORAL, "bold", "ma")
        actions = action_points(t)
        for index, point in enumerate(actions):
            x = action_center[0] + point[0] * 235
            y = action_center[1] - point[1] * 235
            radius = 6 + 1.6 * math.sin(index * 0.6 + t)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=CORAL)

    if reveal(t, 5.15, 0.7) > 0:
        dashed_ellipse(draw, (1440, 340, 1760, 660), CORAL, 5, -t)
        text(draw, (1600, 695), "ACTION FAMILY", 18, CORAL, "bold", "ma")

    if reveal(t, 6.15, 0.7) > 0:
        curved_arrow(draw, (1435, 545), (1300, 685), (1170, 545), "#8793A3", 7, 23)
        text(draw, (1280, 670), "GRADIENT UPDATE", 17, "#77869A", "bold", "mm")

    sphere_ring = reveal(t, 7.35, 0.8)
    if sphere_ring > 0:
        def paint_sphere_ring(ring_draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
            ring_draw.ellipse((710, 280, 1150, 720), outline=(*rgb(BLUE), 235), width=6)
            text(ring_draw, (930, 755), "ACTION-ALIGNED REPRESENTATION", 19, BLUE, "bold", "ma")

        frame = alpha_layer(frame, paint_sphere_ring, sphere_ring)
        draw = ImageDraw.Draw(frame)

    aligned = reveal(t, 8.35, 0.8)
    if aligned > 0:
        draw.line((565, 844, 1355, 844), fill=LINE, width=2)
        text(draw, (960, 900), "ALIGNED", 62, TEAL, "bold", "mm")
        text(draw, (960, 969), "A physically grounded intent becomes a deployable action interface.", 23, MUTED, "regular", "mm")
    return frame


def draw_split_operator(frame: Image.Image, box, compact: bool = False) -> None:
    x0, y0, x1, y1 = map(int, box)
    draw = ImageDraw.Draw(frame)
    offset = max(8, min(18, (x1 - x0) // 16))
    back = (x0 + offset, y0 - offset, x1 + offset, y1 - offset)
    draw.rounded_rectangle(back, radius=20, fill="#FBE8E9", outline=CORAL, width=4)
    rounded_horizontal_gradient(frame, box, 20, "#DCECF8", "#FBE8E9")
    draw = ImageDraw.Draw(frame)
    draw.rounded_rectangle(box, radius=20, outline=BLUE, width=5)
    center_x = (x0 + x1) / 2
    draw_split_name(draw, center_x, y0 + (52 if compact else 76), 30 if compact else 43)
    operator_y = (y0 + y1) / 2 + (4 if compact else 8)
    g_size = 40 if compact else 60
    script_size = 20 if compact else 29
    text(draw, (center_x - (8 if compact else 12), operator_y), "G", g_size, INK, "medium", "mm")
    script_x = center_x + (16 if compact else 22)
    text(draw, (script_x, operator_y - (19 if compact else 28)), "k", script_size, INK, "medium", "mm")
    text(draw, (script_x, operator_y + (19 if compact else 28)), "η", script_size, INK, "medium", "mm")
    text(draw, (center_x, y1 - (34 if compact else 52)), "shared twice", 17 if compact else 24, MUTED, "bold", "mm")


def draw_intact_demo_card(frame: Image.Image, box, current: Image.Image) -> None:
    x0, y0, x1, _ = map(int, box)
    paste_cover(frame, current, box)
    draw = ImageDraw.Draw(frame)
    draw.rectangle(box, outline=CORAL, width=4)
    draw.rectangle((x0, y0, x1, y0 + 44), fill=NIGHT)
    text(draw, ((x0 + x1) / 2, y0 + 23), "INTACT · DIRECT", 17, CORAL, "bold", "mm")


def draw_task_card(
    frame: Image.Image,
    box,
    task: str,
    current: Image.Image,
    goal: Image.Image,
    accent: str,
) -> None:
    x0, y0, x1, y1 = map(int, box)
    draw = ImageDraw.Draw(frame)
    rounded(draw, box, 12, WHITE, "#AEBAC9", 3)
    draw.rectangle((x0, y0, x1, y0 + 9), fill=accent)
    title_size = 19 if x1 - x0 < 250 else 24
    text(draw, (x0 + 16, y0 + 38), task, title_size, INK, "bold")
    pad = 14
    gap = 9
    image_y0 = y0 + 58
    image_y1 = y1 - 38
    image_width = (x1 - x0 - 2 * pad - gap) / 2
    current_box = (x0 + pad, image_y0, x0 + pad + image_width, image_y1)
    goal_box = (x0 + pad + image_width + gap, image_y0, x1 - pad, image_y1)
    paste_cover(frame, current, current_box)
    paste_cover(frame, goal, goal_box)
    draw = ImageDraw.Draw(frame)
    draw.rectangle(current_box, outline=BLUE, width=3)
    draw.rectangle(goal_box, outline=accent, width=3)
    text(draw, ((current_box[0] + current_box[2]) / 2, y1 - 17), "oₜ", 16, MUTED, "medium", "mm")
    text(draw, ((goal_box[0] + goal_box[2]) / 2, y1 - 17), "o_g", 16, MUTED, "medium", "mm")


def scene_figure1(t: float, pairs: list[RolloutPair], include_operator: bool = True) -> Image.Image:
    migration = smooth(t / 2.2)
    paper_amount = smooth((t - 0.25) / 1.55)
    background = blend(base(NIGHT), base(PAPER), paper_amount)
    samples = [(pair.at(1.0)[1], pair.at(1.0)[2]) for pair in pairs]
    starts = ((572, 205, 902, 535), (1496, 205, 1826, 535), (572, 630, 902, 960), (1496, 630, 1826, 960))
    targets = ((48, 248, 268, 492), (286, 248, 506, 492), (48, 526, 268, 770), (286, 526, 506, 770))
    single_frame = background.copy()
    paired_frame = background.copy()
    for pair, sample, start, target, accent in zip(pairs, samples, starts, targets, TASK_COLORS):
        box = tuple(lerp(source, destination, migration) for source, destination in zip(start, target))
        draw_intact_demo_card(single_frame, box, sample[0])
        draw_task_card(paired_frame, box, pair.task, sample[0], sample[1], accent)

    card_morph = reveal(t, 0.45, 1.2)
    frame = blend(single_frame, paired_frame, card_morph)

    draw = ImageDraw.Draw(frame)
    header_alpha = reveal(t, 1.45, 0.75)
    if header_alpha > 0:
        def paint_header(header_draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
            colored_text_runs(
                header_draw,
                (86, 65),
                (
                    ("ALIGN ", INK),
                    ("IN", BLUE),
                    (" TO ", INK),
                    ("ACT", CORAL),
                    (". LEARN BETTER REPRESENTATIONS.", INK),
                ),
                43,
                "bold",
            )
            header_draw.line((86, 137, 1834, 137), fill=LINE, width=2)

        frame = alpha_layer(frame, paint_header, header_alpha)
        draw = ImageDraw.Draw(frame)

    encoder_alpha = reveal(t, 2.1, 0.75)
    if encoder_alpha > 0:
        rounded(draw, (560, 250, 820, 770), 18, "#E1F0FA", BLUE, 5)
        text(draw, (690, 314), "SHARED", 24, BLUE, "bold", "mm")
        text(draw, (690, 362), "VISUAL ENCODER", 27, INK, "bold", "mm")
        text(draw, (674, 482), "E", 78, BLUE, "medium", "mm")
        text(draw, (716, 515), "θ", 38, BLUE, "medium", "mm")
        text(draw, (690, 606), "one latent space", 21, MUTED, "bold", "mm")
        for index, accent in enumerate(TASK_COLORS):
            cx = 625 + index * 44
            draw.rounded_rectangle((cx - 17, 665, cx + 17, 703), radius=5, fill=accent)
            text(draw, (cx, 684), str(index + 1), 18, WHITE, "bold", "mm")
        if t > 2.75:
            arrow(draw, (518, 510), (548, 510), BLUE, 7, 18)

    family_alpha = reveal(t, 3.15, 0.8)
    if family_alpha > 0:
        raw_latent, latent_groups = latent_points(t)
        morph = smooth((t - 6.4) / 1.15)
        sphere = spherical_latent_points(t, len(raw_latent))
        latent = raw_latent * (1.0 - morph) + sphere * morph
        latent_center = (1080, 480)
        action_center = (1660, 480)
        text(draw, (1080, 220), "LATENT INTENT SPACE", 20, BLUE, "bold", "ma")
        text(draw, (1660, 220), "ACTION SPACE", 20, CORAL, "bold", "ma")
        arrow(draw, (830, 505), (870, 505), BLUE, 7, 18)
        for index, point in enumerate(latent):
            x = latent_center[0] + point[0] * 250
            y = latent_center[1] - point[1] * 250
            radius = 5.0 + 1.0 * math.sin(index * 0.4 + t)
            color = BLUE_BRIGHT if latent_groups[index] == 1 else BLUE
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        actions = action_points(t)
        for index, point in enumerate(actions):
            x = action_center[0] + point[0] * 250
            y = action_center[1] - point[1] * 250
            radius = 6.0 + 1.3 * math.sin(index * 0.55 + t)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=CORAL)

        ring_fade = 1.0 - smooth((t - 6.25) / 0.9)
        if ring_fade > 0:
            def paint_intent_ring(ring_draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
                dashed_ellipse(ring_draw, (915, 320, 1245, 650), BLUE, 5, t)
                text(ring_draw, (1080, 692), "INTENT FAMILY", 17, BLUE, "bold", "ma")

            frame = alpha_layer(frame, paint_intent_ring, ring_fade)
            draw = ImageDraw.Draw(frame)
        dashed_ellipse(draw, (1450, 320, 1870, 650), CORAL, 5, -t)
        text(draw, (1660, 692), "ACTION FAMILY", 17, CORAL, "bold", "ma")

    operator_alpha = reveal(t, 4.15, 0.75)
    if operator_alpha > 0:
        curved_arrow(draw, (1260, 445), (1365, 330), (1450, 445), INK, 8, 24)
        text(draw, (1348, 345), "G", 30, INK, "medium", "mm")
        text(draw, (1370, 356), "η", 19, INK, "medium", "mm")

    gradient_alpha = reveal(t, 5.1, 0.75)
    if gradient_alpha > 0:
        curved_arrow(draw, (1450, 555), (1365, 675), (1260, 555), "#8793A3", 7, 22)
        text(draw, (1360, 661), "GRADIENT UPDATE", 16, "#77869A", "bold", "mm")

    sphere_ring = reveal(t, 6.65, 0.7)
    if sphere_ring > 0:
        draw.ellipse((900, 300, 1260, 660), outline=BLUE, width=6)
        text(draw, (1080, 707), "ACTION-ALIGNED REPRESENTATION", 17, BLUE, "bold", "ma")

    aligned_alpha = reveal(t, 7.35, 0.75)
    if aligned_alpha > 0:
        draw.line((1000, 752, 1720, 752), fill=LINE, width=2)
        text(draw, (1360, 807), "ALIGNED", 55, TEAL, "bold", "mm")
        text(draw, (1360, 867), "The intent geometry now predicts a physically grounded action family.", 22, MUTED, "regular", "mm")
    return frame


def draw_condition_panel(draw: ImageDraw.ImageDraw, box, label: str, color: str, lines: tuple[str, str]) -> None:
    rounded(draw, box, 13, WHITE, color, 4)
    x0, y0, x1, _ = box
    text(draw, (x0 + 24, y0 + 34), label, 18, color, "bold")
    text(draw, ((x0 + x1) / 2, y0 + 88), lines[0], 21, INK, "mono", "mm")
    text(draw, ((x0 + x1) / 2, y0 + 135), lines[1], 20, INK, "mono", "mm")


def scene_equivalence_final(t: float) -> Image.Image:
    frame = base(PAPER)
    draw = ImageDraw.Draw(frame)
    text(draw, (86, 67), "THE QUOTIENT VIEW", 43, INK, "bold")
    draw.line((86, 137, 1834, 137), fill=LINE, width=2)

    rounded(draw, (95, 260, 695, 450), 13, WHITE, BLUE, 4)
    text(draw, (120, 297), "ONE SHARED INPUT GRAMMAR", 18, BLUE, "bold")
    text(draw, (395, 360), "hₜ(m) = [ zₜ ; m ; zₜ⊙m ;", 21, INK, "mono", "mm")
    text(draw, (395, 407), "A(aₜ₋₁) ]", 21, INK, "mono", "mm")
    rounded(draw, (95, 530, 370, 705), 12, "#E2F0FA", BLUE, 4)
    text(draw, (120, 563), "LOCAL", 18, BLUE, "bold")
    text(draw, (232, 613), "m = zₜ₊₁ − zₜ", 20, INK, "medium", "mm")
    text(draw, (232, 661), "physical reference", 16, BLUE, "bold", "mm")
    rounded(draw, (420, 530, 695, 705), 12, "#FCE6E7", CORAL, 4)
    text(draw, (445, 563), "GOAL", 18, CORAL, "bold")
    text(draw, (557, 613), "m = sg(z_g) − zₜ", 20, INK, "medium", "mm")
    text(draw, (557, 661), "deployable intent", 16, CORAL, "bold", "mm")
    dashed_rectangle(draw, (65, 215, 725, 760), BLUE, 5, 22, 14)
    text(draw, (395, 195), "CONDITIONAL INTENT CLASS  [y]_z", 20, BLUE, "bold", "ms")

    draw_split_operator(frame, (805, 315, 1125, 675), compact=False)
    draw = ImageDraw.Draw(frame)
    arrow(draw, (740, 487), (790, 487), BLUE, 7, 20)
    arrow(draw, (1148, 487), (1245, 487), INK, 7, 22)

    center = (1535, 487)
    rng = np.random.default_rng(203)
    for group, offset, color in ((0, (-34, -15), BLUE), (1, (36, 17), CORAL)):
        for index in range(34):
            angle = index * 2.399 + t * (0.15 if group == 0 else -0.12)
            radius = 18 + 48 * ((index % 11) / 10)
            x = center[0] + offset[0] + math.cos(angle) * radius + rng.normal(0, 3)
            y = center[1] + offset[1] + math.sin(angle) * radius * 0.58 + rng.normal(0, 2)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    draw.ellipse((center[0] - 13, center[1] - 13, center[0] + 13, center[1] + 13), fill=INK, outline=WHITE, width=4)
    text(draw, (center[0], center[1] - 44), "expert aₜ", 18, INK, "bold", "ms")
    text(draw, (center[0] - 125, center[1] - 112), "local prediction", 17, BLUE, "bold", "mm")
    text(draw, (center[0] + 128, center[1] + 112), "goal prediction", 17, CORAL, "bold", "mm")
    dashed_ellipse(draw, (1285, 300, 1785, 675), CORAL, 6, -t)
    text(draw, (1535, 720), "ACTION-LAW CLASS  Phi_z([y]_z)", 20, CORAL, "bold", "ma")
    text(draw, (1535, 764), "local and goal NLLs share the same expert aₜ", 18, MUTED, "medium", "ma")

    rounded(draw, (355, 875, 1565, 1000), 12, "#EDF3F9", "#C8D4E1", 2)
    text(draw, (960, 916), "EQUAL ACTION LAW  ≠  EQUAL LATENT VECTORS", 26, INK, "bold", "mm")
    text(draw, (960, 965), "A supported action family absorbs small latent drift during direct rollout.", 21, MUTED, "regular", "mm")
    return frame


def scene_isomorphism(t: float, pairs: list[RolloutPair]) -> Image.Image:
    focus = smooth(t / 1.05)
    frame = blend(scene_figure1(10.0, pairs), base(PAPER), focus)
    operator_progress = reveal(t, 0.9, 1.2)

    def paint_retained_operator(draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
        text(draw, (1348, 345), "G", 30, INK, "medium", "mm")
        text(draw, (1370, 356), "η", 19, INK, "medium", "mm")

    retained_opacity = clamp(1.0 - 2.0 * operator_progress)
    frame = alpha_layer(frame, paint_retained_operator, retained_opacity)
    source = (1268, 290, 1448, 500)
    target = (765, 245, 1095, 720)
    operator_box = tuple(lerp(start, end, operator_progress) for start, end in zip(source, target))
    operator_frame = frame.copy()
    draw_split_operator(operator_frame, operator_box, compact=operator_progress < 0.45)
    frame = blend(frame, operator_frame, operator_progress)
    draw = ImageDraw.Draw(frame)

    phase_alpha = reveal(t, 1.95, 0.75)
    if phase_alpha > 0:
        colored_text_runs(
            draw,
            (86, 67),
            (
                ("ISOMORPHIC INPUTS. ", INK),
                ("GOAL", GOAL_GREEN),
                (" AND ", INK),
                ("FUTURE", LOCAL_ORANGE),
                (" SELF-SUPERVISION.", INK),
            ),
            43,
            "bold",
        )
        draw.line((86, 137, 1834, 137), fill=LINE, width=2)

        grammar_progress = reveal(t, 2.2, 0.95)
        local_progress = reveal(t, 2.85, 0.95)
        goal_progress = reveal(t, 3.45, 0.95)

        grammar_x = lerp(-600, 70, grammar_progress)
        rounded(draw, (grammar_x, 405, grammar_x + 580, 540), 12, BLUE_PALE, "#9DC6E3", 4)
        text(draw, (grammar_x + 290, 438), "SAME INTACT INPUT GRAMMAR", 20, "#215E8E", "bold", "mm")
        text(draw, (grammar_x + 290, 493), "[zₜᵏ; mₜ; zₜᵏ ⊙ mₜ; A(aₜ₋₁ᵏ)]", 27, INK, "math", "mm")
        grammar_end = lerp(650, 752, grammar_progress)
        arrow(draw, (650, 472), (grammar_end, 472), BLUE, 7, 20)

        local_x = lerp(-600, 70, local_progress)
        rounded(draw, (local_x, 205, local_x + 580, 325), 12, LOCAL_PALE, LOCAL_ORANGE, 4)
        text(draw, (local_x + 290, 237), "LOCAL · attached", 20, "#A14820", "bold", "mm")
        text(draw, (local_x + 290, 286), "mₜ local,k = zₜ₊₁ᵏ − zₜᵏ", 26, "#A14820", "medium", "mm")
        local_end = lerp(650, 752, local_progress)
        arrow(draw, (650, 265), (local_end, 265), LOCAL_ORANGE, 7, 20)

        goal_x = lerp(-600, 70, goal_progress)
        rounded(draw, (goal_x, 620, goal_x + 580, 740), 12, GOAL_PALE, GOAL_GREEN, 4)
        text(draw, (goal_x + 290, 652), "GOAL · detached", 20, "#11695D", "bold", "mm")
        text(draw, (goal_x + 290, 701), "mₜ goal,k = sg(z_gᵏ) − zₜᵏ", 26, "#11695D", "medium", "mm")
        goal_end = lerp(650, 752, goal_progress)
        arrow(draw, (650, 680), (goal_end, 680), GOAL_GREEN, 7, 20)

    local_output = reveal(t, 4.15, 0.85)
    if local_output > 0:
        output_end = lerp(1118, 1242, local_output)
        arrow(draw, (1118, 265), (output_end, 265), LOCAL_ORANGE, 7, 20)
        rounded(draw, (1258, 215, 1465, 315), 11, LOCAL_PALE, LOCAL_ORANGE, 4)
        text(draw, (1361, 249), "LOCAL OUTPUT", 19, "#A14820", "bold", "mm")
        text(draw, (1361, 286), "âₜˡᵒᶜᵃˡ", 28, "#A14820", "medium", "mm")

    goal_output = reveal(t, 4.75, 0.85)
    if goal_output > 0:
        output_end = lerp(1118, 1242, goal_output)
        arrow(draw, (1118, 680), (output_end, 680), GOAL_GREEN, 7, 20)
        rounded(draw, (1258, 630, 1465, 730), 11, GOAL_PALE, GOAL_GREEN, 4)
        text(draw, (1361, 664), "GOAL OUTPUT", 19, "#11695D", "bold", "mm")
        text(draw, (1361, 701), "âₜᵍᵒᵃˡ", 28, "#11695D", "medium", "mm")

    nll_alpha = reveal(t, 5.4, 0.8)
    if nll_alpha > 0:
        for y, color, pale in ((265, LOCAL_ORANGE, LOCAL_PALE), (680, GOAL_GREEN, GOAL_PALE)):
            arrow(draw, (1480, y), (1630, y), color, 6, 18)
            text(draw, (1555, y - 31), "NLL", 22, CORAL, "bold", "mm")
            rounded(draw, (1645, y - 50, 1815, y + 50), 11, pale, color, 4)
            text(draw, (1730, y - 14), "EXPERT", 15, color, "bold", "mm")
            text(draw, (1730, y + 24), "aₜ", 30, INK, "medium", "mm")

    equivalence_alpha = reveal(t, 6.75, 0.9)
    if equivalence_alpha > 0:
        def paint_equivalence(eq_draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
            text(eq_draw, (960, 792), "ACTION EQUIVALENCE CLASS · WITHIN PREDICTION TOLERANCE", 20, CORAL, "bold", "mm")
            dashed_rectangle(eq_draw, (285, 820, 1635, 938), CORAL, 5, 20, 13)
            chips = ((600, "âₜ local"), (960, "aₜ expert"), (1320, "âₜ goal"))
            for x, label in chips:
                eq_draw.rounded_rectangle((x - 118, 844, x + 118, 914), radius=9, fill="#FFF6F6", outline=CORAL, width=4)
                text(eq_draw, (x, 879), label, 27, INK, "bold", "mm")
            text(eq_draw, (780, 879), "≈", 34, CORAL, "bold", "mm")
            text(eq_draw, (1140, 879), "≈", 34, CORAL, "bold", "mm")
            text(eq_draw, (960, 992), "A supported action class improves rollout robustness and mitigates drift.", 27, INK, "bold", "mm")

        frame = alpha_layer(frame, paint_equivalence, equivalence_alpha)
    return frame


def scene_bars(t: float) -> Image.Image:
    frame = base(PAPER)
    draw = ImageDraw.Draw(frame)
    text(draw, (86, 67), "CONTROL KNOWLEDGE, EXPOSED DIRECTLY", 43, INK, "bold")
    draw.line((86, 137, 1834, 137), fill=LINE, width=2)

    chart = (120, 235, 1460, 850)
    x0, y0, x1, y1 = chart
    for value in (0, 25, 50, 75, 100):
        y = y1 - value / 100 * (y1 - y0)
        draw.line((x0, y, x1, y), fill=LINE, width=2)
        text(draw, (x0 - 25, y), str(value), 17, MUTED, "regular", "rm")
    text(draw, (x0, y0 - 40), "SR (%)", 18, MUTED, "bold")

    rise = smooth((t - 0.65) / 3.1)
    group_w = (x1 - x0) / 4
    for index, (task, baseline, direct, guarded) in enumerate(zip(TASKS, LEWM_SR, INTACT_SR, GUARDED_SR)):
        center = x0 + (index + 0.5) * group_w
        bars = ((-78, baseline, "#8390A2"), (0, direct, BLUE), (78, guarded, CORAL))
        for offset, value, color in bars:
            bar_w = 60
            height = value / 100 * (y1 - y0) * rise
            box = (center + offset - bar_w / 2, y1 - height, center + offset + bar_w / 2, y1)
            draw.rounded_rectangle(box, radius=6, fill=color)
            if rise > 0.82:
                text(draw, (center + offset, y1 - height - 16), f"{value:.1f}", 18, color, "bold", "ms")
        text(draw, (center, y1 + 40), task, 20, INK, "bold", "ma")

    legend_y = 972
    legends = ((180, "#8390A2", "LeWM · CEM 300×30"), (535, BLUE, "INTACT · Direct · 0"), (860, CORAL, "INTACT · Guarded A · 384"))
    for x, color, label in legends:
        draw.rectangle((x, legend_y - 12, x + 34, legend_y + 12), fill=color)
        text(draw, (x + 50, legend_y), label, 18, MUTED, "medium", "lm")

    panel_alpha = reveal(t, 3.1, 0.8)
    if panel_alpha > 0:
        draw.line((1530, 225, 1530, 870), fill=LINE, width=2)
        text(draw, (1690, 255), "MACRO SR", 20, MUTED, "bold", "ma")
        metrics = (
            ("LeWM", LEWM_MACRO, "9,000 seq.", "#8390A2"),
            ("Direct", INTACT_MACRO, "0 seq.", BLUE),
            ("Guarded A", GUARDED_MACRO, "384 seq.", CORAL),
        )
        for index, (label, value, budget, color) in enumerate(metrics):
            y = 350 + index * 185
            text(draw, (1585, y - 38), label.upper(), 17, color, "bold")
            text(draw, (1795, y), f"{value:.2f}%", 37, color, "bold", "ra")
            text(draw, (1585, y + 45), budget, 17, MUTED, "medium")
            if index < 2:
                draw.line((1585, y + 92, 1795, y + 92), fill=LINE, width=2)
        rounded(draw, (1575, 843, 1810, 904), 8, "#FCE5E7", "#F4A7AC", 2)
        text(draw, (1692, 873), "+1.53 pp with local verification", 16, CORAL, "bold", "mm")
    return frame


def pearson(points: list[dict]) -> float:
    if len(points) < 3:
        return float("nan")
    return float(np.corrcoef([point["cka"] for point in points], [point["sr"] for point in points])[0, 1])


def scene_training_geometry(
    t: float,
    points: list[dict],
    latent_2d: np.memmap,
    latent_epochs: np.ndarray,
    effective_rank: np.ndarray,
) -> Image.Image:
    frame = base(PAPER)
    draw = ImageDraw.Draw(frame)
    text(draw, (86, 67), "CORRESPONDENCE BECOMES PREDICTIVE", 43, INK, "bold")
    draw.line((86, 137, 1834, 137), fill=LINE, width=2)

    epoch_progress = smooth((t - 0.45) / 7.1)
    current_epoch = lerp(1.0, 5.0, epoch_progress)
    grouped = []
    for epoch in range(1, 6):
        rows = [point for point in points if point["epoch"] == epoch]
        grouped.append((epoch, float(np.mean([row["sr"] for row in rows])), float(np.mean([row["cka"] for row in rows]))))

    chart = (135, 230, 895, 720)
    x0, y0, x1, y1 = chart
    draw.line((x0, y0, x0, y1), fill=INK, width=3)
    draw.line((x0, y1, x1, y1), fill=INK, width=3)
    for value in (25, 50, 75, 100):
        y = y1 - (value - 25) / 75 * (y1 - y0)
        draw.line((x0, y, x1, y), fill=LINE, width=2)
        text(draw, (x0 - 18, y), str(value), 16, CORAL, "medium", "rm")
    for value in (0.42, 0.46, 0.50, 0.54):
        y = y1 - (value - 0.42) / 0.12 * (y1 - y0)
        text(draw, (x1 + 18, y), f"{value:.2f}", 15, BLUE, "medium", "lm")
    text(draw, (x0, 195), "DIRECT MACRO SR (%)", 17, CORAL, "bold")
    text(draw, (x1, 195), "MEAN LINEAR CKA", 17, BLUE, "bold", "ra")

    def point_xy(epoch: float, value: float, metric: str) -> tuple[float, float]:
        x = x0 + (epoch - 1.0) / 4.0 * (x1 - x0)
        if metric == "sr":
            y = y1 - clamp((value - 25.0) / 75.0) * (y1 - y0)
        else:
            y = y1 - clamp((value - 0.42) / 0.12) * (y1 - y0)
        return x, y

    current_sr = float(np.interp(current_epoch, [item[0] for item in grouped], [item[1] for item in grouped]))
    current_cka = float(np.interp(current_epoch, [item[0] for item in grouped], [item[2] for item in grouped]))
    for metric, color, values in (("sr", CORAL, [item[1] for item in grouped]), ("cka", BLUE, [item[2] for item in grouped])):
        path = []
        for epoch, value in zip(range(1, 6), values):
            if epoch <= current_epoch:
                path.append(point_xy(epoch, value, metric))
        if current_epoch < 5.0:
            interpolated = current_sr if metric == "sr" else current_cka
            path.append(point_xy(current_epoch, interpolated, metric))
        if len(path) > 1:
            draw.line(path, fill=color, width=8, joint="curve")
        for epoch, value in zip(range(1, 6), values):
            x, y = point_xy(epoch, value, metric)
            radius = 10 if epoch <= current_epoch + 0.01 else 7
            fill = color if epoch <= current_epoch + 0.01 else PAPER
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=color, width=4)
    for epoch in range(1, 6):
        x = x0 + (epoch - 1) / 4 * (x1 - x0)
        text(draw, (x, y1 + 40), f"E{epoch}", 17, INK if epoch <= current_epoch + 0.01 else MUTED, "bold", "ma")

    frame_index = int(np.argmin(np.abs(latent_epochs - current_epoch)))
    cloud = np.asarray(latent_2d[frame_index], dtype=np.float32).copy()
    cloud -= np.mean(cloud, axis=0, keepdims=True)
    radii = np.linalg.norm(cloud, axis=1)
    robust_radius = max(float(np.quantile(radii, 0.98)), 1.0)
    cloud /= robust_radius
    center = (1455, 530)
    cloud_scale = 285
    draw.line((1050, center[1], 1850, center[1]), fill="#DCE3EC", width=2)
    draw.line((center[0], 175, center[0], 830), fill="#DCE3EC", width=2)
    text(draw, (1050, 160), "SYNCHRONIZED 2D LATENT GEOMETRY", 19, INK, "bold")
    text(draw, (1850, 160), f"E{current_epoch:.2f}", 26, PURPLE, "bold", "ra")
    order = np.argsort(np.linalg.norm(cloud, axis=1))[::-1]
    for index in order:
        task_index = min(3, index // 96)
        x = center[0] + cloud[index, 0] * cloud_scale
        y = center[1] - cloud[index, 1] * cloud_scale
        color = TASK_COLORS[task_index]
        radius = 5.2 + 0.7 * math.sin(index * 0.37 + t * 1.4)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE, width=1)
    legend_x = 1110
    for task, color in zip(TASKS, TASK_COLORS):
        draw.ellipse((legend_x - 6, 872, legend_x + 6, 884), fill=color)
        text(draw, (legend_x + 15, 878), task, 15, INK, "bold", "lm")
        legend_x += 185

    rank = float(effective_rank[frame_index])
    r_value = pearson(points)
    tiles = (
        ("TRAINING", f"E{current_epoch:.2f}", PURPLE),
        ("MACRO SR", f"{current_sr:.2f}%", CORAL),
        ("MEAN CKA", f"{current_cka:.3f}", BLUE),
        ("EFFECTIVE RANK", f"{rank:.1f}", TEAL),
    )
    for index, (label, value, color) in enumerate(tiles):
        x = 135 + index * 195
        rounded(draw, (x, 800, x + 180, 915), 8, WHITE, LINE, 2)
        text(draw, (x + 90, 832), label, 15, MUTED, "bold", "mm")
        text(draw, (x + 90, 880), value, 28, color, "bold", "mm")
    rounded(draw, (135, 935, 900, 1025), 8, "#EEEAFD", "#C9BDF5", 2)
    text(draw, (285, 963), "PREDICTED-EXPERT ACTION CKA", 17, PURPLE, "bold", "mm")
    arrow(draw, (450, 963), (610, 963), PURPLE, 5, 16)
    arrow(draw, (610, 963), (450, 963), PURPLE, 5, 16)
    text(draw, (755, 963), "DIRECT MACRO SR", 18, PURPLE, "bold", "mm")
    text(draw, (517, 1004), f"15 CHECKPOINTS · PEARSON r = {r_value:+.3f}", 18, PURPLE, "bold", "mm")
    text(draw, (1455, 982), "Fixed image identities across every training epoch", 17, MUTED, "medium", "mm")
    return frame


def scene_correspondence(t: float, points: list[dict]) -> Image.Image:
    frame = base(PAPER)
    draw = ImageDraw.Draw(frame)
    text(draw, (86, 67), "CORRESPONDENCE BECOMES PREDICTIVE", 43, INK, "bold")
    draw.line((86, 137, 1834, 137), fill=LINE, width=2)

    epoch = 1 if t < 0.82 else min(5, 2 + int(max(0.0, t - 0.82) / 1.12))
    visible = [point for point in points if point["epoch"] <= epoch]
    current = [point for point in points if point["epoch"] == epoch]
    chart = (145, 245, 1325, 835)
    x0, y0, x1, y1 = chart
    for value, label in ((25, "25"), (50, "50"), (75, "75"), (100, "100")):
        y = y1 - (value - 25) / 75 * (y1 - y0)
        draw.line((x0, y, x1, y), fill=LINE, width=2)
        text(draw, (x0 - 24, y), label, 17, MUTED, "regular", "rm")
    draw.line((x0, y0, x0, y1), fill=INK, width=3)
    draw.line((x0, y1, x1, y1), fill=INK, width=3)
    for value, label in ((0.40, "0.40"), (0.48, "0.48"), (0.56, "0.56")):
        x = x0 + (value - 0.40) / 0.16 * (x1 - x0)
        draw.line((x, y1, x, y1 + 10), fill=INK, width=2)
        text(draw, (x, y1 + 35), label, 17, MUTED, "regular", "ma")
    text(draw, ((x0 + x1) / 2, 920), "Predicted / expert linear CKA", 20, INK, "medium", "ma")
    text(draw, (x0, 205), "Direct macro SR (%)", 20, INK, "medium")

    def map_point(point: dict) -> tuple[float, float]:
        x = x0 + clamp((point["cka"] - 0.40) / 0.16) * (x1 - x0)
        y = y1 - clamp((point["sr"] - 25.0) / 75.0) * (y1 - y0)
        return x, y

    xs = np.asarray([point["cka"] for point in visible])
    ys = np.asarray([point["sr"] for point in visible])
    if len(visible) >= 3 and float(np.var(xs)) > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        low = max(0.40, float(xs.min()) - 0.004)
        high = min(0.56, float(xs.max()) + 0.004)
        start = map_point({"cka": low, "sr": slope * low + intercept})
        end = map_point({"cka": high, "sr": slope * high + intercept})
        draw.line((*start, *end), fill=PURPLE, width=7)

    epoch_colors = (PURPLE, TEAL, GOLD, CORAL, BLUE)
    for point in visible:
        x, y = map_point(point)
        radius = 14 if point["epoch"] == epoch else 9
        color = epoch_colors[point["epoch"] - 1]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE, width=3)

    panel_x = 1450
    text(draw, (panel_x, 250), "THROUGH EPOCH", 18, MUTED, "bold")
    text(draw, (panel_x, 334), f"E{epoch}", 68, INK, "bold")
    mean_sr = float(np.mean([point["sr"] for point in current]))
    mean_cka = float(np.mean([point["cka"] for point in current]))
    r_value = pearson(visible)
    draw.line((panel_x, 405, 1810, 405), fill=LINE, width=2)
    text(draw, (panel_x, 468), "MACRO SR", 17, MUTED, "bold")
    text(draw, (1810, 468), f"{mean_sr:.2f}%", 34, CORAL, "bold", "ra")
    text(draw, (panel_x, 557), "MEAN CKA", 17, MUTED, "bold")
    text(draw, (1810, 557), f"{mean_cka:.3f}", 34, BLUE, "bold", "ra")
    text(draw, (panel_x, 646), "PEARSON r", 17, MUTED, "bold")
    text(draw, (1810, 646), f"{r_value:+.3f}", 34, PURPLE, "bold", "ra")
    text(draw, (panel_x, 706), f"cumulative n = {len(visible)}", 17, MUTED, "regular")
    for index in range(1, 6):
        x = panel_x + (index - 1) * 75
        draw.line((x, 805, x + 48, 805), fill=epoch_colors[index - 1] if index <= epoch else LINE, width=8)
        text(draw, (x + 24, 842), f"E{index}", 15, INK if index <= epoch else MUTED, "bold", "mm")
    text(draw, (960, 1010), "Each epoch adds three independently trained checkpoints.", 18, MUTED, "regular", "mm")
    return frame


def scene_headlines(
    t: float,
    points: np.ndarray,
    labels: np.ndarray,
    logo: Image.Image,
    institution_marks: tuple[tuple[str, Image.Image], ...] = (),
) -> Image.Image:
    frame = base(NIGHT)
    frame = draw_particle_space(frame, points, labels, t + 7.5, center=(430, 540), scale=455, opacity=0.52)
    draw = ImageDraw.Draw(frame)
    text(draw, (86, 67), "THE WORLD MODEL ALREADY KNOWS", 21, "#EDF3FC", "medium")
    draw.line((86, 117, 1834, 117), fill="#20324A", width=2)

    draw_film_overline(draw, 430, 350, 340, 210, 18)
    mark = fit_rgba(logo, 660, 235)
    frame = frame.convert("RGBA")
    frame.alpha_composite(mark, (430 - mark.width // 2, 540 - mark.height // 2))
    frame = frame.convert("RGB")
    draw = ImageDraw.Draw(frame)

    stats = (
        ("ONLY", "1 EPOCH"),
        ("TRAINING BUDGET", "1 / 10"),
        ("INFERENCE TIME", "≈ 1 / 300"),
        ("SR · RELATIVE", "+11.2%"),
    )
    tiles = ((900, 180), (1370, 180), (900, 350), (1370, 350))
    for index, ((label, value), (x, y)) in enumerate(zip(stats, tiles)):
        tile_alpha = reveal(t, 0.35 + index * 0.5, 0.42)

        def paint_tile(
            layer_draw: ImageDraw.ImageDraw,
            _overlay: Image.Image,
            label=label,
            value=value,
            x=x,
            y=y,
            index=index,
        ) -> None:
            center_x = x + 185
            layer_draw.line((x, y, x + 370, y), fill="#243750", width=2)
            text(layer_draw, (center_x, y + 36), label, 16, "#8EA1BB", "bold", "ma")
            text(layer_draw, (center_x, y + 98), value, 43, CORAL if index in (0, 3) else BLUE_BRIGHT, "bold", "ma")

        frame = alpha_layer(frame, paint_tile, tile_alpha)

    left_x = 1085
    center_x = 1320
    right_x = 1555
    baseline_y = 625
    latency_y = 854

    baseline_progress = reveal(t, 2.32, 0.65)
    if baseline_progress > 0:
        draw = ImageDraw.Draw(frame)
        draw.line((900, baseline_y, lerp(900, 1740, baseline_progress), baseline_y), fill="#28405D", width=5)

    gray_alpha = reveal(t, 3.02, 0.55)

    def paint_gray(layer_draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
        layer_draw.ellipse((left_x - 19, baseline_y - 19, left_x + 19, baseline_y + 19), fill="#8390A2")
        text(layer_draw, (left_x, 680), "LeWM", 18, "#9AAAC0", "bold", "ma")
        text(layer_draw, (left_x, 724), "85.75%", 29, WHITE, "bold", "ma")
        text(layer_draw, (left_x, latency_y - 22), "1.48 s", 34, WHITE, "bold", "mm")
        text(layer_draw, (left_x, latency_y + 22), "CEM 300×30", 16, "#8EA1BB", "medium", "mm")

    frame = alpha_layer(frame, paint_gray, gray_alpha)

    purple_alpha = reveal(t, 3.7, 0.6)

    def paint_purple(layer_draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
        curved_arrow(layer_draw, (left_x + 31, 585), (center_x, 510), (right_x - 39, 585), PURPLE, 7, 24)
        text(layer_draw, (center_x, 528), "+9.58 pp", 25, PURPLE, "bold", "mm")
        arrow(layer_draw, (1210, latency_y), (1430, latency_y), PURPLE, 6, 20)
        text(layer_draw, (center_x, latency_y - 38), "≈300×", 21, PURPLE, "bold", "mm")
        text(layer_draw, (center_x, latency_y + 52), "PLANNER LATENCY", 17, "#8EA1BB", "bold", "mm")

    frame = alpha_layer(frame, paint_purple, purple_alpha)

    red_alpha = reveal(t, 4.35, 0.6)

    def paint_red(layer_draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
        layer_draw.ellipse((right_x - 23, baseline_y - 23, right_x + 23, baseline_y + 23), fill=CORAL, outline=WHITE, width=4)
        text(layer_draw, (right_x, 678), "95.33%", 43, CORAL, "bold", "ma")
        text(layer_draw, (right_x, 726), "DIRECT · 0 search", 17, CORAL, "bold", "ma")
        text(layer_draw, (right_x, latency_y - 22), "2.9–5.5 ms", 34, CORAL, "bold", "mm")
        text(layer_draw, (right_x, latency_y + 22), "INTACT Direct", 16, CORAL, "medium", "mm")

    frame = alpha_layer(frame, paint_red, red_alpha)
    draw = ImageDraw.Draw(frame)

    footer_alpha = reveal(t, 5.05, 0.6)

    def paint_footer(layer_draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
        text(layer_draw, (430, 742), "Learn the interface. Keep the world model.", 30, WHITE, "medium", "mm")

    frame = alpha_layer(frame, paint_footer, footer_alpha)

    loop_fade = reveal(t, 6.2, 0.9)
    if loop_fade > 0:
        return blend(frame, scene_opening(0.0, points, labels, logo, institution_marks), loop_fade)
    return frame


def make_qr_code(value: str, module_size: int = 10, quiet_zone: int = 4) -> Image.Image:
    """Build a standards-compliant QR code without adding a release dependency."""
    encoded = cv2.QRCodeEncoder_create().encode(value)
    encoded = np.pad(encoded, quiet_zone, mode="constant", constant_values=255)
    size = encoded.shape[0] * module_size
    return Image.fromarray(encoded).convert("RGB").resize((size, size), Image.Resampling.NEAREST)


def monochrome_mark(image: Image.Image, remove_white_background: bool = False) -> Image.Image:
    pixels = np.asarray(image.convert("RGBA")).copy()
    if remove_white_background:
        foreground = 255 - np.min(pixels[:, :, :3], axis=2)
        background_alpha = np.clip(foreground.astype(np.float32) / 42.0, 0.0, 1.0)
        pixels[:, :, 3] = (pixels[:, :, 3].astype(np.float32) * background_alpha).astype(np.uint8)
    pixels[:, :, :3] = rgb("#EAF1FA")
    mark = Image.fromarray(pixels)
    bounds = mark.getchannel("A").getbbox()
    return mark.crop(bounds) if bounds else mark


def load_institution_marks(assets: Path) -> tuple[tuple[str, Image.Image], ...]:
    return (
        (
            "ZHEJIANG UNIVERSITY",
            monochrome_mark(Image.open(assets / "zhejiang-university-logo.png"), remove_white_background=True),
        ),
        (
            "TSINGHUA UNIVERSITY",
            monochrome_mark(Image.open(assets / "tsinghua-university-logo.jpg"), remove_white_background=True),
        ),
        ("INSPATIO", monochrome_mark(Image.open(assets / "inspatio-logo.png"))),
        ("ROBOPARTY LAB", monochrome_mark(Image.open(assets / "roboparty-lab-logo.png"))),
    )


def draw_institution_row(
    frame: Image.Image,
    institution_marks: tuple[tuple[str, Image.Image], ...],
    centers: tuple[int, ...],
    mark_y: int,
    name_y: int,
    max_width: int,
    max_height: int,
    name_size: int,
) -> Image.Image:
    for center_x, (name, institution_mark) in zip(centers, institution_marks):
        fitted = fit_rgba(institution_mark, max_width, max_height)
        frame = frame.convert("RGBA")
        frame.alpha_composite(fitted, (center_x - fitted.width // 2, mark_y - fitted.height // 2))
        frame = frame.convert("RGB")
        draw = ImageDraw.Draw(frame)
        text(draw, (center_x, name_y), name, name_size, "#AFC0D5", "medium", "ma")
    return frame


def scene_share_card(
    t: float,
    points: np.ndarray,
    labels: np.ndarray,
    logo: Image.Image,
    qr_code: Image.Image,
    institution_marks: tuple[tuple[str, Image.Image], ...],
) -> Image.Image:
    closing = scene_headlines(5.9, points, labels, logo, institution_marks)
    frame = base(NIGHT)
    frame = draw_particle_space(frame, points, labels, t + 13.4, center=(430, 540), scale=455, opacity=0.52)

    def left_veil(draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
        draw.rectangle((985, 0, WIDTH, HEIGHT), fill=(*rgb(NIGHT), 82))

    frame = alpha_layer(frame, left_veil)
    mark = fit_rgba(logo, 660, 235)
    frame = frame.convert("RGBA")
    frame.alpha_composite(mark, (430 - mark.width // 2, 540 - mark.height // 2))
    frame = frame.convert("RGB")
    draw = ImageDraw.Draw(frame)
    text(draw, (86, 67), "THE WORLD MODEL ALREADY KNOWS", 21, "#EDF3FC", "medium")
    draw.line((86, 117, 1834, 117), fill="#20324A", width=2)
    draw_film_overline(draw, 430, 350, 340, 210, 18)
    text(draw, (430, 742), "Learn the interface. Keep the world model.", 30, WHITE, "medium", "mm")
    clean_frame = frame.copy()

    text(draw, (85, 852), "AFFILIATIONS", 14, "#8295AF", "bold")
    frame = draw_institution_row(
        frame,
        institution_marks,
        centers=(155, 385, 615, 845),
        mark_y=918,
        name_y=982,
        max_width=176,
        max_height=64,
        name_size=13,
    )

    def paint_card(card_draw: ImageDraw.ImageDraw, overlay: Image.Image) -> None:
        card_draw.rounded_rectangle((1197, 205, 1713, 721), radius=8, fill=CORAL)
        card_draw.rounded_rectangle((1179, 187, 1695, 703), radius=8, fill=BLUE_BRIGHT)
        card_draw.rounded_rectangle((1188, 196, 1704, 712), radius=8, fill=WHITE)
        qr_x = 1188 + (516 - qr_code.width) // 2
        qr_y = 196 + (516 - qr_code.height) // 2
        overlay.paste(qr_code.convert("RGBA"), (qr_x, qr_y))

    frame = alpha_layer(frame, paint_card)

    def paint_details(detail_draw: ImageDraw.ImageDraw, _overlay: Image.Image) -> None:
        detail_draw.line((1100, 752, 1790, 752), fill="#263A54", width=2)
        detail_rows = (
            (790, "PROJECT PAGE", PROJECT_URL, BLUE_BRIGHT),
            (840, "GITHUB", GITHUB_URL, FILM_RED),
            (890, "INSPATIO", INSPATIO_URL, "#C7D3E3"),
            (940, "ROBOPARTY LAB", ROBOPARTY_URL, "#C7D3E3"),
        )
        for y, label, value, color in detail_rows:
            text(detail_draw, (1100, y), label, 15, color, "bold", "lm")
            text(detail_draw, (1790, y), value, 19, WHITE, "medium", "rm")
        text(detail_draw, (1446, 1009), "SCAN TO OPEN THE PROJECT PAGE", 22, WHITE, "bold", "mm")

    frame = alpha_layer(frame, paint_details)
    cleared = blend(closing, clean_frame, reveal(t, 0.05, 0.75))
    return blend(cleared, frame, reveal(t, 0.72, 0.85))


def write_video(path: Path, frames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-g",
        str(FPS * 2),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            assert process.stdin is not None
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        if process.stdin:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {path}")


def build_share_film(
    assets: Path,
    points: np.ndarray,
    labels: np.ndarray,
    logo: Image.Image,
    institution_marks: tuple[tuple[str, Image.Image], ...],
) -> None:
    hero = assets / "intact-hero-film.mp4"
    if not hero.exists():
        raise FileNotFoundError(f"Build the standard project film first: {hero}")

    qr_code = make_qr_code(PROJECT_URL)
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(np.asarray(qr_code))
    if decoded != PROJECT_URL:
        raise RuntimeError(f"Generated QR code did not validate: {decoded!r}")

    build_dir = assets.parent.parent / ".build/share-film"
    build_dir.mkdir(parents=True, exist_ok=True)
    for index, timestamp in enumerate((0.0, 0.75, 1.6, 4.8), start=1):
        scene_share_card(timestamp, points, labels, logo, qr_code, institution_marks).save(
            build_dir / f"share-card-{index:02d}.jpg",
            quality=96,
            subsampling=0,
        )

    outro_duration = 6.2
    outro = build_dir / "share-outro.mp4"
    write_video(
        outro,
        (
            scene_share_card(index / FPS, points, labels, logo, qr_code, institution_marks)
            for index in range(round(outro_duration * FPS))
        ),
    )

    share_cut = 60.3
    destination = assets / "intact-project-film-share.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(hero),
            "-i",
            str(outro),
            "-filter_complex",
            (
                f"[0:v]trim=duration={share_cut},setpts=PTS-STARTPTS[main];"
                "[1:v]setpts=PTS-STARTPTS[outro];"
                "[main][outro]concat=n=2:v=1:a=0[video]"
            ),
            "-map",
            "[video]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-g",
            str(FPS * 2),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )

    capture = cv2.VideoCapture(str(destination))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - FPS))
    ok, final_frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not decode the share-edition verification frame")
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(final_frame)
    if decoded != PROJECT_URL:
        raise RuntimeError(f"Encoded share-edition QR code did not validate: {decoded!r}")
    downsampled = cv2.resize(final_frame, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA)
    decoded_small, _, _ = cv2.QRCodeDetector().detectAndDecode(downsampled)
    if decoded_small != PROJECT_URL:
        raise RuntimeError(f"Downsampled share-edition QR code did not validate: {decoded_small!r}")
    print(
        f"Built {destination.name} with a verified QR code for {PROJECT_URL} "
        f"({share_cut + outro_duration:.1f}s total; {outro_duration:.1f}s share card)"
    )


def load_rollouts(experiment: Path) -> list[RolloutPair]:
    lewm_root = experiment / "recovery_INTACT_eval_fleet/results/eval_staging/lewm_s42/epoch_5/cache"
    intact_root = experiment / "recovery_INTACT_eval_fleet_delta/results/eval_staging/delta_condition_full_s3072/epoch_5/cache"
    pairs = []
    for task, key in zip(TASKS, TASK_KEYS):
        episode = EPISODES[key]
        lewm = read_video(lewm_root / f"recovery_lewm_{key}_s42/env_{episode}.mp4")
        intact = read_video(intact_root / f"recovery_delta_full_{key}_s3072/env_{episode}.mp4")
        pairs.append(RolloutPair(task, key, episode, lewm, intact))
    return pairs


def build_film(
    assets: Path,
    pairs: list[RolloutPair],
    points: np.ndarray,
    labels: np.ndarray,
    correlation_points: list[dict],
    latent_2d: np.memmap,
    latent_epochs: np.ndarray,
    effective_rank: np.ndarray,
    preview_only: bool,
    share_only: bool,
) -> None:
    logo = film_logo(Image.open(assets / "intact-wordmark-light.png").convert("RGBA"))
    institution_marks = load_institution_marks(assets)
    if share_only:
        build_share_film(assets, points, labels, logo, institution_marks)
        return

    scenes = [
        (5.4, lambda t: scene_opening(t, points, labels, logo, institution_marks)),
        (10.0, lambda t: scene_demos(t, pairs)),
        (10.0, lambda t: scene_figure1(t, pairs)),
        (12.0, lambda t: scene_isomorphism(t, pairs)),
        (9.0, lambda t: scene_training_geometry(t, correlation_points, latent_2d, latent_epochs, effective_rank)),
        (8.0, scene_bars),
        (7.5, lambda t: scene_headlines(t, points, labels, logo, institution_marks)),
    ]
    transition = 0.6

    def render(global_t: float) -> Image.Image:
        cursor = 0.0
        for index, (duration, renderer) in enumerate(scenes):
            if global_t < cursor + duration or index == len(scenes) - 1:
                local_t = global_t - cursor
                frame = renderer(local_t)
                if local_t > duration - transition:
                    next_renderer = scenes[(index + 1) % len(scenes)][1]
                    next_frame = next_renderer(0.0)
                    return blend(frame, next_frame, (local_t - duration + transition) / transition)
                return frame
            cursor += duration
        raise AssertionError("unreachable")

    preview_dir = assets.parent.parent / ".build/hero-film-storyboard"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_times = (
        2.8,
        8.4,
        14.2,
        15.2,
        15.38,
        15.4,
        15.6,
        20.0,
        22.0,
        23.8,
        25.2,
        25.38,
        25.4,
        25.6,
        25.9,
        26.4,
        26.9,
        27.4,
        28.0,
        30.0,
        34.8,
        39.0,
        45.6,
        48.2,
        53.2,
        55.2,
        56.2,
        57.2,
        58.0,
        58.9,
        59.9,
        60.8,
    )
    for index, timestamp in enumerate(preview_times, start=1):
        render(timestamp).save(preview_dir / f"{index:02d}.jpg", quality=95, subsampling=0)

    poster = scene_opening(2.8, points, labels, logo, institution_marks)
    poster.save(assets / "intact-hero-film-poster.jpg", quality=96, subsampling=0, optimize=True)
    if preview_only:
        print(f"Built storyboard in {preview_dir}")
        return

    total = sum(duration for duration, _ in scenes)
    write_video(
        assets / "intact-hero-film.mp4",
        (render(index / FPS) for index in range(round(total * FPS))),
    )
    print(f"Built {total:.1f}s INTACT film at {WIDTH}×{HEIGHT}, {FPS} fps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--research-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview-only", action="store_true")
    mode.add_argument("--share-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    experiment = args.research_root.resolve() / "inverse_lewm_arxiv/experiments"
    assets = repo / "docs/assets"
    pairs = load_rollouts(experiment)
    points, labels = load_particle_cloud(assets / "latent-geometry")
    latent_2d, latent_epochs, effective_rank = load_latent_movie(assets / "latent-geometry")
    correlation = json.loads((assets / "goal-intact-alignment.json").read_text())["points"]
    build_film(
        assets,
        pairs,
        points,
        labels,
        correlation,
        latent_2d,
        latent_epochs,
        effective_rank,
        args.preview_only,
        args.share_only,
    )


if __name__ == "__main__":
    main()
