#!/usr/bin/env python3
"""Export privacy-safe E5 PCA coordinates for the website action viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TASKS = (
    ("pusht", "PushT", "#7152D9"),
    ("cube", "Cube", "#17A88E"),
    ("reacher", "Reacher", "#E7A33B"),
    ("tworoom", "TwoRoom", "#EA5A60"),
)


def balanced_indices(length: int, count: int) -> np.ndarray:
    if length <= count:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, count, dtype=np.int64)


def fit_pca(values: np.ndarray, dimensions: int = 3):
    values = values.astype(np.float64)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-7)
    standardized = (values - mean) / scale
    center = standardized.mean(axis=0)
    _, _, vectors = np.linalg.svd(standardized - center, full_matrices=False)
    components = vectors[:dimensions]

    def transform(samples: np.ndarray) -> np.ndarray:
        return ((samples.astype(np.float64) - mean) / scale - center) @ components.T

    return transform


def normalize(*chunks: np.ndarray) -> tuple[np.ndarray, ...]:
    combined = np.concatenate(chunks, axis=0)
    center = np.median(combined, axis=0)
    centered = [chunk - center for chunk in chunks]
    radii = np.linalg.norm(np.concatenate(centered, axis=0), axis=1)
    scale = max(float(np.quantile(radii, 0.985)), 1e-7)
    return tuple(chunk / scale for chunk in centered)


def rounded(values: np.ndarray) -> list[list[float]]:
    return np.round(values, 5).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Directory containing one E5 folder per task")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intent-points", type=int, default=240)
    parser.add_argument("--action-pairs", type=int, default=160)
    args = parser.parse_args()

    arrays = {}
    metadata = {}
    for key, _, _ in TASKS:
        directory = args.input / key
        with np.load(directory / "alignment_samples.npz", allow_pickle=False) as payload:
            arrays[key] = {name: payload[name].astype(np.float32) for name in payload.files}
        metadata[key] = json.loads((directory / "metadata.json").read_text())

    intent_samples = {}
    for key, _, _ in TASKS:
        values = arrays[key]["actor_intent"]
        intent_samples[key] = values[balanced_indices(len(values), args.intent_points)]
    intent_transform = fit_pca(np.concatenate(list(intent_samples.values()), axis=0))
    projected_intents = [intent_transform(intent_samples[key]) for key, _, _ in TASKS]
    projected_intents = normalize(*projected_intents)

    output_tasks = []
    for (key, label, color), intent in zip(TASKS, projected_intents):
        payload = arrays[key]
        indices = balanced_indices(len(payload["expert_action"]), args.action_pairs)
        expert = payload["expert_action"][indices]
        predicted = payload["predicted_action"][indices]
        action_transform = fit_pca(np.concatenate((expert, predicted), axis=0))
        expert_3d, predicted_3d = normalize(action_transform(expert), action_transform(predicted))
        metrics = metadata[key]["metrics"]
        output_tasks.append(
            {
                "key": key,
                "label": label,
                "color": color,
                "intent": rounded(intent),
                "expert": rounded(expert_3d),
                "predicted": rounded(predicted_3d),
                "metrics": {
                    "action_r2": round(float(metrics["expert_action_r2"]), 4),
                    "linear_cka": round(float(metrics["predicted_expert_cka"]), 4),
                    "knn_overlap": round(float(metrics["predicted_expert_knn_overlap"]), 4),
                },
            }
        )

    export = {
        "title": "INTACT E5 action correspondence",
        "epoch": 5,
        "projection": "standardized PCA-3D with robust display normalization",
        "raw_latents_included": False,
        "tasks": output_tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export, separators=(",", ":")))
    print(f"Exported {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
