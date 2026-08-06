import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def select_episode_fraction(dataset, fraction: float, seed: int):
    """Restrict a trajectory dataset to a deterministic nested episode subset."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"data_fraction must be in (0, 1], got {fraction}")

    lengths = np.asarray(dataset.lengths)
    eligible = np.flatnonzero(lengths >= int(dataset.span)).astype(np.int64)
    if eligible.size == 0:
        raise ValueError("dataset has no episode long enough for one training clip")

    rng = np.random.default_rng(seed)
    ordered = rng.permutation(eligible)
    selected_count = max(1, int(np.floor(eligible.size * fraction)))
    selected = np.sort(ordered[:selected_count])
    selected_set = set(selected.tolist())

    full_clip_count = len(dataset.clip_indices)
    if selected_count < eligible.size:
        dataset.clip_indices = [
            (episode, start)
            for episode, start in dataset.clip_indices
            if episode in selected_set
        ]

    digest = hashlib.sha256(selected.tobytes()).hexdigest()
    info = {
        "fraction": float(fraction),
        "subset_seed": int(seed),
        "eligible_episodes": int(eligible.size),
        "selected_episodes": int(selected_count),
        "full_clips": int(full_clip_count),
        "selected_clips": int(len(dataset.clip_indices)),
        "episode_ids_sha256": digest,
    }
    return selected, info


def build_episode_split(lengths, seed: int, ratios=(0.8, 0.1, 0.1)):
    """Build a deterministic train/validation/test split over whole episodes."""
    lengths = np.asarray(lengths, dtype=np.int64)
    ratios = np.asarray(ratios, dtype=np.float64)
    if lengths.ndim != 1 or lengths.size == 0:
        raise ValueError("episode lengths must be a non-empty one-dimensional array")
    if ratios.shape != (3,) or np.any(ratios <= 0.0):
        raise ValueError(f"split ratios must contain three positive values, got {ratios}")
    ratios = ratios / ratios.sum()

    ordered = np.random.default_rng(seed).permutation(lengths.size).astype(np.int64)
    train_end = int(np.floor(lengths.size * ratios[0]))
    val_end = train_end + int(np.floor(lengths.size * ratios[1]))
    splits = {
        "train": np.sort(ordered[:train_end]),
        "val": np.sort(ordered[train_end:val_end]),
        "test": np.sort(ordered[val_end:]),
    }
    if any(values.size == 0 for values in splits.values()):
        raise ValueError(
            f"episode split produced an empty partition for {lengths.size} episodes"
        )
    return splits


def _int64_sha256(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def make_episode_split_manifest(
    dataset,
    *,
    dataset_name: str,
    seed: int,
    ratios=(0.8, 0.1, 0.1),
):
    """Create a serializable manifest tied to the dataset episode structure."""
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    splits = build_episode_split(lengths, seed=seed, ratios=ratios)
    normalized_ratios = np.asarray(ratios, dtype=np.float64)
    normalized_ratios = normalized_ratios / normalized_ratios.sum()
    return {
        "version": 1,
        "dataset_name": str(dataset_name),
        "seed": int(seed),
        "ratios": {
            "train": float(normalized_ratios[0]),
            "val": float(normalized_ratios[1]),
            "test": float(normalized_ratios[2]),
        },
        "num_episodes": int(lengths.size),
        "episode_lengths_sha256": _int64_sha256(lengths),
        "splits": {
            name: [int(value) for value in values]
            for name, values in splits.items()
        },
        "split_sha256": {
            name: _int64_sha256(values) for name, values in splits.items()
        },
    }


def save_episode_split_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_episode_split_manifest(path, dataset=None):
    """Load a split manifest and optionally validate it against a dataset."""
    path = Path(path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    required = {
        "version",
        "dataset_name",
        "seed",
        "num_episodes",
        "episode_lengths_sha256",
        "splits",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"episode split manifest is missing keys: {sorted(missing)}")

    split_arrays = {
        name: np.asarray(manifest["splits"][name], dtype=np.int64)
        for name in ("train", "val", "test")
    }
    manifest["split_sha256"] = {
        name: manifest.get("split_sha256", {}).get(name, _int64_sha256(values))
        for name, values in split_arrays.items()
    }
    combined = np.concatenate(list(split_arrays.values()))
    expected = np.arange(int(manifest["num_episodes"]), dtype=np.int64)
    if combined.size != expected.size or not np.array_equal(np.sort(combined), expected):
        raise ValueError("episode split manifest is not a disjoint full partition")

    if dataset is not None:
        lengths = np.asarray(dataset.lengths, dtype=np.int64)
        if lengths.size != int(manifest["num_episodes"]):
            raise ValueError(
                "episode split dataset size mismatch: "
                f"manifest={manifest['num_episodes']} dataset={lengths.size}"
            )
        digest = _int64_sha256(lengths)
        if digest != manifest["episode_lengths_sha256"]:
            raise ValueError("episode split dataset length signature mismatch")

    manifest["path"] = str(path)
    manifest["split_arrays"] = split_arrays
    return manifest


def episode_column_values(dataset, internal_episode_ids):
    """Map internal episode indices to values stored in the row episode column."""
    episode_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    stored_values = np.unique(np.asarray(dataset.get_col_data(episode_col)))
    if stored_values.size != len(dataset.lengths):
        raise ValueError(
            "dataset episode column does not match the episode structure: "
            f"column={stored_values.size} lengths={len(dataset.lengths)}"
        )
    internal_episode_ids = np.asarray(internal_episode_ids, dtype=np.int64)
    return stored_values[internal_episode_ids]


def select_episode_rows(dataset, values, internal_episode_ids):
    """Select flat rows belonging to internal episode indices."""
    values = np.asarray(values)
    internal_episode_ids = np.asarray(internal_episode_ids, dtype=np.int64)
    if internal_episode_ids.size == 0:
        raise ValueError("cannot select rows from an empty episode set")
    if internal_episode_ids.min() < 0 or internal_episode_ids.max() >= len(dataset.lengths):
        raise ValueError("internal episode index is outside the dataset")

    mask = np.zeros(values.shape[0], dtype=bool)
    offsets = np.asarray(dataset.offsets, dtype=np.int64)
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    for episode in internal_episode_ids:
        start = int(offsets[episode])
        end = start + int(lengths[episode])
        mask[start:end] = True
    return values[mask]


def get_column_normalizer(
    dataset,
    source: str,
    target: str,
    episode_ids: np.ndarray | None = None,
):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    if episode_ids is not None:
        col_data = select_episode_rows(dataset, col_data, episode_ids)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
