from types import SimpleNamespace

import pytest
import torch

from jepa import JEPA
from module import Embedder, IntentActionActor
from train import construct_intents, paired_intent_action_loss


class CountingActor(IntentActionActor):
    def __init__(self):
        super().__init__(
            embed_dim=3,
            action_emb_dim=3,
            action_dim=2,
            hidden_dim=16,
            depth=1,
        )
        self.calls = 0

    def forward(self, z, intent, prev_act_emb):
        self.calls += 1
        return super().forward(z, intent, prev_act_emb)


def make_cfg(local_start=0, goal_start=0, local_weight=0.1, goal_weight=0.05):
    return SimpleNamespace(
        loss=SimpleNamespace(
            intent=SimpleNamespace(
                local_weight=local_weight,
                goal_weight=goal_weight,
                local_start=local_start,
                goal_start=goal_start,
            )
        )
    )


def make_model(actor):
    return JEPA(
        encoder=torch.nn.Identity(),
        predictor=torch.nn.Identity(),
        action_encoder=Embedder(input_dim=2, emb_dim=3),
        intent_actor=actor,
        intent_mode="goal_displacement",
    )


def test_paired_objective_uses_two_calls_through_one_actor():
    actor = CountingActor()
    embeddings = torch.randn(4, 8, 3, requires_grad=True)
    actions = torch.randn(4, 8, 2)
    previous_actions = torch.randn(4, 8, 2)
    result = paired_intent_action_loss(
        make_model(actor), embeddings, actions, previous_actions, make_cfg()
    )
    assert actor.calls == 2
    assert result["action_loss"].ndim == 0
    result["action_loss"].backward()
    assert embeddings.grad is not None


@pytest.mark.parametrize(
    ("local_weight", "goal_weight", "expected_calls"),
    [(0.0, 0.0, 0), (0.1, 0.0, 1), (0.0, 0.05, 1), (0.1, 0.05, 2)],
)
def test_disabled_objective_branches_do_not_call_actor(
    local_weight, goal_weight, expected_calls
):
    actor = CountingActor()
    embeddings = torch.randn(2, 8, 3)
    actions = torch.randn(2, 8, 2)
    previous_actions = torch.randn(2, 8, 2)
    result = paired_intent_action_loss(
        make_model(actor),
        embeddings,
        actions,
        previous_actions,
        make_cfg(local_weight=local_weight, goal_weight=goal_weight),
    )
    assert actor.calls == expected_calls
    if expected_calls == 0:
        assert result["action_loss"].item() == 0.0


def test_goal_window_and_detachment_contract():
    embeddings = torch.randn(2, 8, 3, requires_grad=True)
    local, goal = construct_intents(
        embeddings, "goal_displacement", local_start=0, goal_start=0
    )
    assert local.shape == (2, 7, 3)
    assert goal.shape == (2, 7, 3)
    goal.sum().backward()
    assert embeddings.grad[:, :7].abs().sum() > 0
    assert torch.equal(embeddings.grad[:, 7], torch.zeros_like(embeddings.grad[:, 7]))


def test_waypoint_uses_seven_to_one_remaining_steps():
    embeddings = torch.zeros(1, 8, 1)
    embeddings[:, 7] = 10
    _, waypoint = construct_intents(
        embeddings, "waypoint", local_start=0, goal_start=0
    )
    assert waypoint[0, :, 0].tolist() == pytest.approx(
        [10 / 7, 10 / 6, 2.0, 2.5, 10 / 3, 5.0, 10.0]
    )


@pytest.mark.parametrize(
    ("local_start", "goal_start", "expected_local", "expected_goal"),
    [(0, 2, 7, 5), (0, 0, 7, 7), (1, 1, 6, 6)],
)
def test_window_protocol_lengths(
    local_start, goal_start, expected_local, expected_goal
):
    actor = CountingActor()
    embeddings = torch.randn(2, 8, 3)
    actions = torch.randn(2, 8, 2)
    previous_actions = torch.randn(2, 8, 2)
    local, goal = construct_intents(
        embeddings,
        "goal_displacement",
        local_start=local_start,
        goal_start=goal_start,
    )
    assert local.size(1) == expected_local
    assert goal.size(1) == expected_goal
    paired_intent_action_loss(
        make_model(actor),
        embeddings,
        actions,
        previous_actions,
        make_cfg(local_start, goal_start),
    )
    assert actor.calls == 2
