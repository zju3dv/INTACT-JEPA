from types import SimpleNamespace

import torch
import pytest
from torch import nn

from jepa import JEPA
from module import Embedder, IntentActionActor


def test_five_slot_grammar_is_exact():
    actor = IntentActionActor(
        embed_dim=4,
        action_emb_dim=3,
        action_dim=2,
        hidden_dim=16,
        depth=1,
        feature_layout="five_slot",
    )
    z = torch.randn(2, 5, 4)
    intent = torch.randn(2, 5, 4)
    previous = torch.randn(2, 5, 3)
    features = actor.actor_features(z, intent, previous)
    assert features.shape == (2, 5, 19)
    assert torch.equal(features[..., :4], z)
    assert torch.equal(features[..., 4:8], intent)
    assert torch.equal(features[..., 8:12], torch.zeros_like(z))
    assert torch.equal(features[..., 12:16], z * intent)
    assert torch.equal(features[..., 16:], previous)


def test_four_slot_grammar_removes_only_constant_zero_slot():
    actor = IntentActionActor(
        embed_dim=4,
        action_emb_dim=3,
        action_dim=2,
        hidden_dim=16,
        depth=1,
        feature_layout="four_slot",
    )
    z = torch.randn(2, 5, 4)
    intent = torch.randn(2, 5, 4)
    previous = torch.randn(2, 5, 3)
    features = actor.actor_features(z, intent, previous)
    assert features.shape == (2, 5, 15)
    assert torch.equal(features[..., :4], z)
    assert torch.equal(features[..., 4:8], intent)
    assert torch.equal(features[..., 8:12], z * intent)
    assert torch.equal(features[..., 12:], previous)


def test_intent_actor_returns_diagonal_gaussian():
    actor = IntentActionActor(
        embed_dim=4, action_emb_dim=3, action_dim=2, hidden_dim=16, depth=2
    )
    mean, log_std = actor(
        torch.randn(7, 4), torch.randn(7, 4), torch.randn(7, 3)
    )
    assert mean.shape == (7, 2)
    assert log_std.shape == (7, 2)
    assert torch.all(log_std >= -5)
    assert torch.all(log_std <= 2)


class DummyEncoder(nn.Module):
    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        latent = pixels.mean(dim=(-2, -1))
        return SimpleNamespace(last_hidden_state=latent[:, None])


class DummyPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, 3, dim))

    def forward(self, latent, action_embedding):
        return latent + 0 * action_embedding[..., : latent.size(-1)]


def build_tiny_model(intent_mode):
    dim = 3
    action_dim = 4
    return JEPA(
        encoder=DummyEncoder(),
        predictor=DummyPredictor(dim),
        action_encoder=Embedder(input_dim=action_dim, emb_dim=dim),
        intent_actor=IntentActionActor(
            embed_dim=dim,
            action_emb_dim=dim,
            action_dim=action_dim,
            hidden_dim=16,
            depth=1,
        ),
        intent_mode=intent_mode,
    )


def test_goal_intent_modes_and_direct_action_shape():
    current = torch.tensor([[1.0, 2.0, 3.0]])
    goal = torch.tensor([[5.0, 6.0, 7.0]])
    goal_model = build_tiny_model("goal_displacement")
    waypoint_model = build_tiny_model("waypoint")
    assert torch.equal(goal_model.goal_intent(current, goal, 4), goal - current)
    assert torch.equal(
        waypoint_model.goal_intent(current, goal, 4), (goal - current) / 4
    )

    info = {
        "pixels": torch.randn(2, 3, 3, 4, 4),
        "goal": torch.randn(2, 1, 3, 4, 4),
        "action": torch.zeros(2, 3, 4),
    }
    actions = goal_model.get_action(info, horizon=5)
    assert actions.shape == (2, 5, 4)
    assert goal_model.last_direct_diagnostics["candidate_sequences"] == 0


def test_direct_action_rejects_missing_previous_action_history():
    model = build_tiny_model("goal_displacement")
    info = {
        "pixels": torch.randn(1, 3, 3, 4, 4),
        "goal": torch.randn(1, 1, 3, 4, 4),
    }
    with pytest.raises(ValueError, match="Action history is required"):
        model.get_action(info, horizon=1)
