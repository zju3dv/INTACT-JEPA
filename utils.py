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

    def __init__(self, mean, std, raw_nan_value=None):
        self.mean = mean
        self.std = std
        self.raw_nan_value = raw_nan_value

    def __call__(self, x):
        if self.raw_nan_value is not None:
            x = torch.nan_to_num(x, nan=float(self.raw_nan_value))
        return ((x - self.mean) / self.std).float()


class RawZeroActionProcessor:
    """Z-score actions while mapping missing history to a raw zero command."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean_ = np.asarray(mean)
        self.scale_ = np.asarray(std)
        if not np.isfinite(self.mean_).all():
            raise ValueError("Action mean must be finite")
        if not np.isfinite(self.scale_).all() or np.any(self.scale_ <= 0):
            raise ValueError("Action standard deviation must be finite and positive")

    @classmethod
    def fit(cls, values: np.ndarray) -> "RawZeroActionProcessor":
        values = np.asarray(values)
        valid = values[~np.isnan(values).any(axis=1)]
        if valid.shape[0] < 2:
            raise ValueError("At least two finite actions are required")
        return cls(valid.mean(axis=0), valid.std(axis=0, ddof=1))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        values = np.where(np.isnan(values), 0.0, values)
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        return values * self.scale_ + self.mean_


def _trajectory_index_source(dataset):
    """Find the dataset that owns clip starts and episode offsets."""
    if hasattr(dataset, "clip_indices") and hasattr(dataset, "offsets"):
        return dataset
    for child in getattr(dataset, "datasets", []):
        try:
            return _trajectory_index_source(child)
        except ValueError:
            continue
    child = getattr(dataset, "dataset", None)
    if child is not None:
        return _trajectory_index_source(child)
    raise ValueError("Dataset does not expose clip_indices and episode offsets")


class PreviousActionDataset(torch.utils.data.Dataset):
    """Attach a boundary-aware previous action chunk to every training clip.

    The source files remain unchanged. Missing primitive actions before an episode
    begins are raw-zero padded and then normalized with statistics fitted only on
    real actions. Interior clips always receive their true preceding action chunk.
    """

    def __init__(self, dataset, action_mean: torch.Tensor, action_std: torch.Tensor):
        self.dataset = dataset
        self.index_source = _trajectory_index_source(dataset)
        self.frameskip = int(self.index_source.frameskip)
        self.action_mean = action_mean.detach().cpu().reshape(1, -1)
        self.action_std = action_std.detach().cpu().reshape(1, -1)
        if not torch.isfinite(self.action_mean).all():
            raise ValueError("Action mean must be finite")
        if not torch.isfinite(self.action_std).all() or torch.any(self.action_std <= 0):
            raise ValueError("Action standard deviation must be finite and positive")
        module_name = type(self.index_source).__module__
        self.action_rows = (
            None
            if module_name.endswith(".formats.lance")
            else np.asarray(dataset.get_col_data("action"))
        )

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # Never proxy pickle/data-model hooks: doing so would serialize the
        # wrapped reader's state in place of this wrapper's own __dict__.
        if name.startswith("__") or name in {
            "dataset",
            "index_source",
            "action_rows",
        }:
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def _load_action_rows(self, rows: list[int]) -> np.ndarray:
        if self.action_rows is not None:
            return np.asarray(self.action_rows[rows])
        return np.asarray(self.dataset.get_row_data(rows)["action"])

    def _history_rows(self, index: int) -> list[int]:
        episode, start = self.index_source.clip_indices[index]
        available = min(int(start), self.frameskip)
        if not available:
            return []
        global_start = int(self.index_source.offsets[episode]) + int(start)
        return list(range(global_start - available, global_start))

    def _preceding_chunk(
        self,
        index: int,
        *,
        dtype: torch.dtype,
        history: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = self._history_rows(index)
        raw = torch.zeros(
            self.frameskip,
            self.action_mean.size(1),
            dtype=self.action_mean.dtype,
        )
        if rows:
            if history is None:
                history = self._load_action_rows(rows)
            history = torch.as_tensor(
                history,
                dtype=self.action_mean.dtype,
            )
            # Some collection formats encode the missing reset action as NaN.
            # In raw coordinates that boundary convention is the neutral command.
            history = torch.nan_to_num(history, nan=0.0)
            raw[-len(rows):] = history
        normalized = (raw - self.action_mean) / self.action_std
        return normalized.reshape(-1).to(dtype=dtype)

    def _attach(
        self,
        index: int,
        item: dict,
        history: np.ndarray | torch.Tensor | None = None,
    ) -> dict:
        action = item.get("action")
        if not torch.is_tensor(action) or action.ndim != 2:
            raise ValueError("Expected normalized action chunks with shape [T,D]")
        if not torch.isfinite(action).all():
            raise ValueError("Target action chunks must be finite after preprocessing")
        first = self._preceding_chunk(
            index, dtype=action.dtype, history=history
        )
        if first.numel() != action.size(-1):
            raise ValueError(
                "Previous action chunk dimension does not match target chunks: "
                f"{first.numel()} != {action.size(-1)}"
            )
        result = dict(item)
        result["previous_action"] = torch.cat(
            [first.view(1, -1), action[:-1]], dim=0
        )
        return result

    def __getitem__(self, index: int) -> dict:
        return self._attach(index, self.dataset[index])

    def __getitems__(self, indices: list[int]) -> list[dict]:
        if hasattr(self.dataset, "__getitems__"):
            items = self.dataset.__getitems__(indices)
        else:
            items = [self.dataset[index] for index in indices]
        row_lists = [self._history_rows(index) for index in indices]
        unique_rows = sorted({row for rows in row_lists for row in rows})
        lookup = {}
        if unique_rows:
            actions = self._load_action_rows(unique_rows)
            lookup = {row: actions[i] for i, row in enumerate(unique_rows)}
        histories = [
            np.stack([lookup[row] for row in rows]) if rows else None
            for rows in row_lists
        ]
        return [
            self._attach(index, item, history)
            for index, item, history in zip(indices, items, histories)
        ]


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


def get_column_stats(
    dataset,
    source: str,
    episode_ids: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute finite-only z-score statistics for a dataset column."""
    col_data = dataset.get_col_data(source)
    if episode_ids is not None:
        col_data = select_episode_rows(dataset, col_data, episode_ids)
    data = np.asarray(col_data)
    invalid = np.isnan(data).any(axis=1)
    if invalid.any():
        data = data[~invalid]
    if data.shape[0] < 2:
        raise ValueError(f"Column {source!r} has fewer than two finite rows")
    # Compute directly in NumPy to avoid a second full-column Torch allocation.
    mean = torch.from_numpy(
        np.asarray(data.mean(axis=0, keepdims=True)).copy()
    )
    std = torch.from_numpy(
        np.asarray(data.std(axis=0, ddof=1, keepdims=True)).copy()
    )
    return mean, std


def get_column_normalizer(
    dataset,
    source: str,
    target: str,
    episode_ids: np.ndarray | None = None,
    stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    raw_nan_value: float | None = None,
):
    """Get normalizer for a specific column in the dataset."""
    mean, std = (
        stats
        if stats is not None
        else get_column_stats(dataset, source, episode_ids)
    )
    return dt.transforms.WrapTorchTransform(
        ZScoreNormalizer(mean, std, raw_nan_value=raw_nan_value),
        source=source,
        target=target,
    )

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch + 1
        should_save = epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs
        if trainer.is_global_zero and should_save:
            self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
