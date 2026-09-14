"""Causal action-history adapter for dataset-initialized evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockStandardScaler:
    """Apply primitive-action statistics to each slot of a flattened block."""

    scaler: object

    def _primitive_dim(self) -> int:
        mean = getattr(self.scaler, "mean_", getattr(self.scaler, "mean", None))
        if mean is None:
            raise ValueError("The wrapped action scaler has no fitted mean")
        return int(np.asarray(mean).size)

    def transform(self, values):
        array = np.asarray(values)
        primitive_dim = self._primitive_dim()
        if array.shape[-1] % primitive_dim:
            raise ValueError(
                f"Action width {array.shape[-1]} is not divisible by {primitive_dim}"
            )
        shape = array.shape
        transformed = self.scaler.transform(array.reshape(-1, primitive_dim))
        return transformed.reshape(shape)

    def inverse_transform(self, values):
        array = np.asarray(values)
        primitive_dim = self._primitive_dim()
        if array.shape[-1] % primitive_dim:
            raise ValueError(
                f"Action width {array.shape[-1]} is not divisible by {primitive_dim}"
            )
        shape = array.shape
        restored = self.scaler.inverse_transform(array.reshape(-1, primitive_dim))
        return restored.reshape(shape)


class StatefulActionHistoryPolicy:
    """Inject pre-start expert actions, then shift in actions actually executed."""

    def __init__(self, policy, initial_history: np.ndarray):
        self.policy = policy
        history = np.asarray(initial_history, dtype=np.float32)
        if history.ndim != 3:
            raise ValueError(f"Expected history [B, slots, action_dim], got {history.shape}")
        self.history = history.copy()
        self.first_call = True
        self.seed = getattr(policy, "seed", None)

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def set_env(self, env):
        return self.policy.set_env(env)

    def set_seed(self, seed):
        self.seed = seed
        setter = getattr(self.policy, "set_seed", None)
        return setter(seed) if setter is not None else None

    def get_action(self, info_dict, **kwargs):
        if self.first_call:
            self.first_call = False
        else:
            latest = np.asarray(info_dict["action"], dtype=np.float32)
            if latest.ndim == 3:
                latest = latest[:, -1]
            if latest.ndim != 2:
                raise ValueError(f"Expected executed actions [B, D], got {latest.shape}")
            self.history = np.concatenate(
                [self.history[:, 1:], latest[:, None]], axis=1
            )

        prepared = dict(info_dict)
        prepared["action"] = self.history.reshape(self.history.shape[0], 1, -1)
        return self.policy.get_action(prepared, **kwargs)


def build_initial_history(dataset, episodes, start_steps, slots: int) -> np.ndarray:
    """Return ``rows[t-slots:t]`` with raw-zero padding before episode start."""
    if slots < 1:
        raise ValueError("Action-history slots must be positive")
    action_dim = int(np.asarray(dataset.get_col_data("action")).shape[-1])
    histories = []
    for episode, start in zip(episodes, start_steps):
        start = int(start)
        history = np.zeros((slots, action_dim), dtype=np.float32)
        if start > 0:
            begin = max(0, start - slots)
            chunks = dataset.load_chunk(
                np.asarray([episode]),
                np.asarray([begin]),
                np.asarray([start]),
            )
            previous = np.asarray(chunks[0]["action"], dtype=np.float32)
            previous = np.nan_to_num(previous, nan=0.0)
            if len(previous) > slots:
                raise ValueError(
                    f"Loaded {len(previous)} previous actions for {slots} slots"
                )
            if len(previous):
                history[-len(previous) :] = previous
        histories.append(history)
    return np.stack(histories)
