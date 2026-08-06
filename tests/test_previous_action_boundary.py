import numpy as np
import torch

from utils import PreviousActionDataset, RawZeroActionProcessor, ZScoreNormalizer


class FakeClipDataset:
    def __init__(self):
        self.frameskip = 3
        self.num_steps = 3
        self.offsets = np.array([0])
        self.lengths = np.array([12])
        self.clip_indices = [(0, 0), (0, 1), (0, 3)]
        self.raw_actions = np.array(
            [[10.0 + i, 20.0 + 2 * i] for i in range(12)], dtype=np.float32
        )
        self.mean = torch.tensor([[10.0, 20.0]])
        self.std = torch.tensor([[2.0, 5.0]])

    def __len__(self):
        return len(self.clip_indices)

    def get_col_data(self, name):
        assert name == "action"
        return self.raw_actions

    def get_row_data(self, rows):
        return {"action": self.raw_actions[rows]}

    def __getitem__(self, index):
        _, start = self.clip_indices[index]
        raw = torch.from_numpy(
            self.raw_actions[start : start + self.frameskip * self.num_steps]
        )
        normalized = (raw - self.mean) / self.std
        return {"action": normalized.reshape(self.num_steps, -1)}

    def __getitems__(self, indices):
        return [self[index] for index in indices]


def make_dataset():
    base = FakeClipDataset()
    return base, PreviousActionDataset(base, base.mean, base.std)


def normalized_raw_zero(base):
    primitive = (torch.zeros_like(base.mean) - base.mean) / base.std
    return primitive.repeat(base.frameskip, 1).reshape(-1)


def test_episode_start_uses_normalized_raw_zero_chunk():
    base, dataset = make_dataset()
    item = dataset[0]
    assert torch.equal(item["previous_action"][0], normalized_raw_zero(base))
    assert torch.equal(item["previous_action"][1:], item["action"][:-1])


def test_early_clip_left_pads_only_missing_primitive_actions():
    base, dataset = make_dataset()
    item = dataset[1]
    expected_raw = torch.cat(
        [torch.zeros(2, 2), torch.from_numpy(base.raw_actions[0:1])], dim=0
    )
    expected = ((expected_raw - base.mean) / base.std).reshape(-1)
    assert torch.equal(item["previous_action"][0], expected)


def test_interior_clip_uses_true_preceding_chunk():
    base, dataset = make_dataset()
    item = dataset[2]
    previous_raw = torch.from_numpy(base.raw_actions[0:3])
    expected = ((previous_raw - base.mean) / base.std).reshape(-1)
    assert torch.equal(item["previous_action"][0], expected)
    assert torch.equal(item["previous_action"][1:], item["action"][:-1])


def test_batched_fetch_preserves_boundary_alignment():
    _, dataset = make_dataset()
    batch = dataset.__getitems__([2, 0, 1])
    assert len(batch) == 3
    for item in batch:
        assert item["previous_action"].shape == item["action"].shape


def test_eval_processor_maps_missing_action_before_normalization():
    processor = RawZeroActionProcessor(
        mean=np.array([2.0, -3.0]), std=np.array([4.0, 2.0])
    )
    transformed = processor.transform(np.array([[np.nan, np.nan]]))
    np.testing.assert_allclose(transformed, np.array([[-0.5, 1.5]]))
    np.testing.assert_allclose(
        processor.inverse_transform(transformed), np.zeros((1, 2))
    )


def test_training_action_normalizer_maps_raw_nan_to_zero_before_zscore():
    normalizer = ZScoreNormalizer(
        mean=torch.tensor([2.0, -3.0]),
        std=torch.tensor([4.0, 2.0]),
        raw_nan_value=0.0,
    )
    transformed = normalizer(torch.tensor([[float("nan"), float("nan")]]))
    torch.testing.assert_close(transformed, torch.tensor([[-0.5, 1.5]]))
