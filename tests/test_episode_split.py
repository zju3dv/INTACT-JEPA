import json
import tempfile
from pathlib import Path

import numpy as np

from utils import (
    build_episode_split,
    episode_column_values,
    load_episode_split_manifest,
    make_episode_split_manifest,
)


class FakeDataset:
    def __init__(self, lengths):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.column_names = ["episode_idx"]
        self._episode_idx = np.repeat(
            np.arange(len(self.lengths), dtype=np.int64), self.lengths
        )

    def get_col_data(self, name):
        assert name == "episode_idx"
        return self._episode_idx


def test_episode_split_is_deterministic_disjoint_and_complete():
    first = build_episode_split(np.arange(1, 101), seed=17)
    second = build_episode_split(np.arange(1, 101), seed=17)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert [len(first[name]) for name in ("train", "val", "test")] == [80, 10, 10]
    combined = np.concatenate(list(first.values()))
    assert np.array_equal(np.sort(combined), np.arange(100))


def test_manifest_round_trip_and_dataset_validation(tmp_path):
    dataset = FakeDataset([8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    manifest = make_episode_split_manifest(
        dataset,
        dataset_name="fake.lance",
        seed=23,
        ratios=(0.8, 0.1, 0.1),
    )
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    loaded = load_episode_split_manifest(path, dataset=dataset)
    assert loaded["num_episodes"] == 10
    assert len(loaded["split_arrays"]["train"]) == 8
    assert episode_column_values(dataset, [0, 3, 9]).tolist() == [0, 3, 9]


if __name__ == "__main__":
    test_episode_split_is_deterministic_disjoint_and_complete()
    with tempfile.TemporaryDirectory() as directory:
        test_manifest_round_trip_and_dataset_validation(Path(directory))
    print("episode split tests passed")
