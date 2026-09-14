from types import SimpleNamespace

import numpy as np
import torch

from train import construct_intents, paired_intent_action_loss
from utils import PreviousActionDataset


class RecordingModel:
    intent_mode = "goal_displacement"

    def __init__(self):
        self.calls = []

    def action_nll(self, **kwargs):
        self.calls.append(kwargs)
        rows = kwargs["z"].shape[0]
        device = kwargs["z"].device
        return {
            "nll": torch.ones(rows, device=device),
            "mae": torch.ones(rows, device=device),
        }


def _cfg():
    return SimpleNamespace(
        loss=SimpleNamespace(
            intent=SimpleNamespace(local_weight=0.1, goal_weight=0.05)
        )
    )


def test_local_and_goal_conditions_each_cover_all_seven_transitions():
    model = RecordingModel()
    embeddings = torch.randn(2, 8, 3)
    previous = torch.randn(2, 7, 10)
    target = torch.randn(2, 7, 10)

    paired_intent_action_loss(model, embeddings, previous, target, _cfg())

    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["z"].shape == (28, 1, 3)
    assert call["previous_action"].shape == (28, 1, 10)
    torch.testing.assert_close(call["target_action"][:14], call["target_action"][14:])


def test_goal_endpoint_is_detached_but_each_current_state_receives_gradient():
    embeddings = torch.randn(2, 8, 3, requires_grad=True)
    _, goal = construct_intents(embeddings, "goal_displacement")
    assert goal.shape == (2, 7, 3)
    goal.sum().backward()
    assert embeddings.grad[:, :7].abs().sum() > 0
    assert torch.equal(embeddings.grad[:, 7], torch.zeros_like(embeddings.grad[:, 7]))


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

    def __getitem__(self, index):
        _, start = self.clip_indices[index]
        raw = torch.from_numpy(
            self.raw_actions[start : start + self.frameskip * self.num_steps]
        )
        normalized = (raw - self.mean) / self.std
        return {"action": normalized.reshape(self.num_steps, -1)}


def test_previous_action_uses_raw_zero_only_at_episode_boundary():
    base = FakeClipDataset()
    dataset = PreviousActionDataset(base, base.mean, base.std)

    start_item = dataset[0]
    normalized_zero = (
        (torch.zeros(base.frameskip, 2) - base.mean) / base.std
    ).reshape(-1)
    torch.testing.assert_close(start_item["previous_action"][0], normalized_zero)

    interior_item = dataset[2]
    expected = (
        (torch.from_numpy(base.raw_actions[:3]) - base.mean) / base.std
    ).reshape(-1)
    torch.testing.assert_close(interior_item["previous_action"][0], expected)
    torch.testing.assert_close(
        interior_item["previous_action"][1:], interior_item["action"][:-1]
    )
