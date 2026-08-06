#!/usr/bin/env python3
"""Export temporally aligned 2D/3D t-SNE trajectories for the project website.

The input is a dense diagnostic run whose ``global_step_*`` directories each
contain one latent distribution and metric file per task. Every output frame is
computed from a real saved snapshot. Coordinates are quantized to signed
16-bit integers for efficient browser delivery; no latent interpolation is
performed by this exporter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


TASKS = ("pusht", "cube", "reacher", "tworoom")
TASK_LABELS = ("PushT", "Cube", "Reacher", "TwoRoom")
TASK_COLORS = ("#6f42d9", "#00a18a", "#ee9a2d", "#e85d69")
STEP_RE = re.compile(r"global_step_(\d+)$")
QUANTIZATION_LIMIT = 30_000


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    description: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("KEY", "LABEL", "DESCRIPTION", "PATH"),
        required=True,
        help="Variant key, display label, short description, and diagnostic directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-task", type=int, default=96)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--projection-workers", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    return parser.parse_args()


def step_of(path: Path) -> int:
    match = STEP_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Cannot parse global step from {path}")
    return int(match.group(1))


def complete_snapshots(root: Path) -> list[Path]:
    paths = sorted(
        (path for path in root.glob("global_step_*") if path.is_dir()),
        key=step_of,
    )
    complete = [
        path
        for path in paths
        if all(
            (path / f"{task}_latent_distribution.npz").is_file()
            and (path / f"{task}_metrics.json").is_file()
            for task in TASKS
        )
    ]
    if len(complete) != len(paths):
        raise RuntimeError(
            f"{root} contains {len(paths)} snapshots but only {len(complete)} are complete"
        )
    if not complete:
        raise RuntimeError(f"No complete snapshots found in {root}")
    return complete


def balanced_indices(frames: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(frames))
    unique = np.unique(frames)
    quota, remainder = divmod(count, len(unique))
    selected: list[np.ndarray] = []
    for position, frame in enumerate(unique):
        candidates = np.flatnonzero(frames == frame)
        take = min(quota + int(position < remainder), len(candidates))
        if take:
            offsets = np.linspace(0, len(candidates) - 1, take, dtype=np.int64)
            selected.append(candidates[offsets])
    return np.concatenate(selected)


def load_snapshot(
    path: Path,
    fixed_indices: dict[str, np.ndarray] | None,
    points_per_task: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: dict[str, np.ndarray] = {}
    metrics: list[dict[str, float]] = []

    for task_index, task in enumerate(TASKS):
        latent_path = path / f"{task}_latent_distribution.npz"
        with np.load(latent_path, allow_pickle=False) as payload:
            task_values = payload["sample_latents"].astype(np.float32)
            frames = payload["sample_frame_index"].astype(np.int16)
        task_indices = (
            balanced_indices(frames, points_per_task)
            if fixed_indices is None
            else fixed_indices[task]
        )
        if int(task_indices.max()) >= len(task_values):
            raise RuntimeError(f"Fixed sample indices exceed {latent_path}")
        values.append(task_values[task_indices])
        labels.append(np.full(len(task_indices), task_index, dtype=np.uint8))
        indices[task] = task_indices
        metrics.append(json.loads((path / f"{task}_metrics.json").read_text()))

    summary = {
        "step": float(np.mean([row["global_step"] for row in metrics])),
        "epoch": float(np.mean([row["epoch"] for row in metrics])),
        "effective_rank": float(np.mean([row["effective_rank"] for row in metrics])),
        "mean_pairwise_cosine": float(
            np.mean([row["mean_pairwise_cosine"] for row in metrics])
        ),
    }
    return np.concatenate(values), np.concatenate(labels), indices, summary


def align_to_previous(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    current_center = current.mean(axis=0, keepdims=True)
    previous_center = previous.mean(axis=0, keepdims=True)
    rotation, _ = orthogonal_procrustes(
        current - current_center,
        previous - previous_center,
    )
    return ((current - current_center) @ rotation).astype(np.float32)


def embed_snapshot(
    index: int,
    reduced: np.ndarray,
    dimensions: int,
    perplexity: float,
    iterations: int,
) -> tuple[int, np.ndarray]:
    embedding = TSNE(
        n_components=dimensions,
        perplexity=min(perplexity, (len(reduced) - 1) / 3.0),
        learning_rate="auto",
        init="pca",
        max_iter=max(iterations, 500),
        method="barnes_hut",
        angle=0.5,
        random_state=20260723,
        n_jobs=1,
    ).fit_transform(reduced).astype(np.float32)
    embedding -= embedding.mean(axis=0, keepdims=True)
    return index, embedding


def build_projection(
    reduced_frames: list[np.ndarray],
    dimensions: int,
    perplexity: float,
    iterations: int,
    key: str,
    workers: int,
) -> np.ndarray:
    embeddings: list[np.ndarray | None] = [None] * len(reduced_frames)
    total = len(reduced_frames)
    completed = 0
    with ProcessPoolExecutor(max_workers=min(workers, total)) as pool:
        futures = [
            pool.submit(
                embed_snapshot,
                index,
                reduced,
                dimensions,
                perplexity,
                iterations,
            )
            for index, reduced in enumerate(reduced_frames)
        ]
        for future in as_completed(futures):
            index, embedding = future.result()
            embeddings[index] = embedding
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == total:
                print(f"[{key}/{dimensions}D] {completed}/{total}", flush=True)

    aligned_embeddings: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for embedding in embeddings:
        if embedding is None:
            raise RuntimeError("Projection worker did not return an embedding")
        if previous is not None:
            embedding = align_to_previous(embedding, previous)
        previous = embedding
        aligned_embeddings.append(embedding)
    return np.stack(aligned_embeddings)


def quantize(embeddings: np.ndarray) -> tuple[np.ndarray, float, float]:
    normalized = np.empty_like(embeddings)
    for index, embedding in enumerate(embeddings):
        centered = embedding - embedding.mean(axis=0, keepdims=True)
        radii = np.linalg.norm(centered, axis=1)
        robust_radius = max(float(np.quantile(radii, 0.98)), 1e-6)
        normalized[index] = centered / robust_radius

    absolute = np.abs(normalized)
    robust_limit = float(np.quantile(absolute, 0.9995))
    hard_limit = float(absolute.max())
    coordinate_limit = max(robust_limit, 1.25)
    clipped = np.clip(normalized, -coordinate_limit, coordinate_limit)
    encoded = np.rint(clipped / coordinate_limit * QUANTIZATION_LIMIT).astype("<i2")
    clipped_fraction = float(np.mean(absolute > coordinate_limit))
    if clipped_fraction > 0.001:
        raise RuntimeError(f"Unexpected quantization clipping fraction: {clipped_fraction}")
    return encoded, hard_limit, coordinate_limit


def export_run(
    run: Run,
    output: Path,
    points_per_task: int,
    perplexity: float,
    iterations: int,
    projection_workers: int,
    threads_per_worker: int,
) -> dict[str, object]:
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(threads_per_worker)

    snapshots = complete_snapshots(run.path)
    final_values, task_labels, fixed_indices, _ = load_snapshot(
        snapshots[-1], None, points_per_task
    )
    scaler = StandardScaler().fit(final_values)
    components = min(50, final_values.shape[1], len(final_values) - 1)
    pca = PCA(n_components=components, random_state=20260723).fit(
        scaler.transform(final_values)
    )

    reduced_frames: list[np.ndarray] = []
    summaries: list[dict[str, float]] = []
    for index, path in enumerate(snapshots):
        values, labels, _, summary = load_snapshot(path, fixed_indices, points_per_task)
        if not np.array_equal(labels, task_labels):
            raise RuntimeError(f"Task sample order changed in {path}")
        reduced_frames.append(pca.transform(scaler.transform(values)).astype(np.float32))
        summaries.append(summary)
        if index == 0 or (index + 1) % 500 == 0 or index + 1 == len(snapshots):
            print(f"[{run.key}/load] {index + 1}/{len(snapshots)}", flush=True)

    variant_dir = output / run.key
    variant_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for dimensions in (2, 3):
        embeddings = build_projection(
            reduced_frames,
            dimensions,
            perplexity,
            iterations,
            run.key,
            projection_workers,
        )
        encoded, hard_limit, coordinate_limit = quantize(embeddings)
        filename = f"coordinates-{dimensions}d.i16"
        encoded.tofile(variant_dir / filename)
        files[f"{dimensions}d"] = {
            "file": f"{run.key}/{filename}",
            "dimensions": dimensions,
            "encoding": "little-endian-int16",
            "quantization_limit": QUANTIZATION_LIMIT,
            "hard_coordinate_limit_before_clipping": round(hard_limit, 6),
            "coordinate_limit": round(coordinate_limit, 6),
            "display_normalization": "per-frame-98th-percentile-radius",
            "bytes": int(encoded.nbytes),
        }

    metadata = {
        "key": run.key,
        "label": run.label,
        "description": run.description,
        "frames": len(snapshots),
        "points": int(len(task_labels)),
        "points_per_task": points_per_task,
        "first_step": int(summaries[0]["step"]),
        "final_step": int(summaries[-1]["step"]),
        "final_epoch": round(summaries[-1]["epoch"], 5),
        "steps": [int(row["step"]) for row in summaries],
        "epochs": [round(row["epoch"], 5) for row in summaries],
        "effective_rank": [round(row["effective_rank"], 5) for row in summaries],
        "mean_pairwise_cosine": [
            round(row["mean_pairwise_cosine"], 6) for row in summaries
        ],
        "files": files,
    }
    (variant_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n"
    )
    print(f"[{run.key}] export complete", flush=True)
    return {
        "key": run.key,
        "label": run.label,
        "description": run.description,
        "metadata": f"{run.key}/metadata.json",
        "frames": len(snapshots),
        "final_epoch": metadata["final_epoch"],
        "final_effective_rank": metadata["effective_rank"][-1],
        "final_mean_pairwise_cosine": metadata["mean_pairwise_cosine"][-1],
    }


def main() -> None:
    args = parse_args()
    runs = [Run(key, label, description, Path(path)) for key, label, description, path in args.run]
    keys = [run.key for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Run keys must be unique")
    args.output.mkdir(parents=True, exist_ok=True)

    variants: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(runs))) as pool:
        futures = {
            pool.submit(
                export_run,
                run,
                args.output,
                args.points_per_task,
                args.perplexity,
                args.iterations,
                args.projection_workers,
                args.threads_per_worker,
            ): run.key
            for run in runs
        }
        for future in as_completed(futures):
            result = future.result()
            variants[str(result["key"])] = result

    manifest = {
        "title": "Unified-task JEPA representation geometry",
        "projection": "Fixed observations; final-frame StandardScaler and PCA-50; independently PCA-initialized, Procrustes-aligned t-SNE; robust per-frame display normalization",
        "no_latent_interpolation": True,
        "tasks": [
            {"key": key, "label": label, "color": color}
            for key, label, color in zip(TASKS, TASK_LABELS, TASK_COLORS, strict=True)
        ],
        "variants": [variants[run.key] for run in runs],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    )


if __name__ == "__main__":
    main()
