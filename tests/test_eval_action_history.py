import numpy as np
from types import SimpleNamespace

from clear_eval import causal_history_evaluation
from history_policy import (
    BlockStandardScaler,
    StatefulActionHistoryPolicy,
    build_initial_history,
)


class _Dataset:
    def __init__(self):
        self.actions = np.arange(20, dtype=np.float32).reshape(10, 2)

    def get_col_data(self, name):
        assert name == "action"
        return self.actions

    def load_chunk(self, episodes, starts, ends):
        assert episodes.shape == starts.shape == ends.shape == (1,)
        return [{"action": self.actions[int(starts[0]) : int(ends[0])]}]


class _Scaler:
    mean_ = np.array([2.0, 4.0], dtype=np.float32)

    def transform(self, values):
        return (values - self.mean_) / 2.0

    def inverse_transform(self, values):
        return values * 2.0 + self.mean_


class _Policy:
    def __init__(self):
        self.inputs = []

    def get_action(self, info_dict, **kwargs):
        self.inputs.append(np.asarray(info_dict["action"]).copy())
        return np.zeros((1, 2), dtype=np.float32)


def test_continuation_uses_only_rows_before_start():
    dataset = _Dataset()
    history = build_initial_history(dataset, [0], [7], 5)
    np.testing.assert_array_equal(history[0], dataset.actions[2:7])
    assert not np.any(np.all(history[0] == dataset.actions[7], axis=-1))


def test_episode_prefix_is_left_padded_with_raw_zero():
    dataset = _Dataset()
    history = build_initial_history(dataset, [0], [2], 5)
    np.testing.assert_array_equal(history[0, :3], np.zeros((3, 2)))
    np.testing.assert_array_equal(history[0, 3:], dataset.actions[:2])


def test_policy_shifts_in_executed_action_after_first_call():
    initial = np.arange(10, dtype=np.float32).reshape(1, 5, 2)
    inner = _Policy()
    policy = StatefulActionHistoryPolicy(inner, initial)
    policy.get_action({"action": np.full((1, 1, 2), 999.0, dtype=np.float32)})
    np.testing.assert_array_equal(inner.inputs[0].reshape(1, 5, 2), initial)

    executed = np.array([[[31.0, 32.0]]], dtype=np.float32)
    policy.get_action({"action": executed})
    shifted = inner.inputs[1].reshape(1, 5, 2)
    np.testing.assert_array_equal(shifted[:, :-1], initial[:, 1:])
    np.testing.assert_array_equal(shifted[:, -1], executed[:, -1])


def test_block_scaler_normalizes_each_primitive_slot():
    scaler = BlockStandardScaler(_Scaler())
    raw = np.array([[0.0, 0.0, 2.0, 4.0, 4.0, 8.0]], dtype=np.float32)
    expected = np.array([[-1.0, -2.0, 0.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    normalized = scaler.transform(raw)
    np.testing.assert_allclose(normalized, expected)
    np.testing.assert_allclose(scaler.inverse_transform(normalized), raw)


def test_clear_adapter_reads_world_model_policy_cfg(monkeypatch):
    import stable_worldmodel as swm

    dataset = _Dataset()
    inner = _Policy()
    inner.cfg = SimpleNamespace(action_block=5)
    inner.process = {"action": _Scaler()}
    world = SimpleNamespace(policy=inner)

    def fake_evaluate(bound_world, *args, **kwargs):
        assert isinstance(bound_world.policy, StatefulActionHistoryPolicy)
        assert isinstance(bound_world.policy.policy.process["action"], BlockStandardScaler)
        return {"success_rate": 100.0}

    monkeypatch.setattr(swm.World, "evaluate", fake_evaluate)
    with causal_history_evaluation("direct"):
        result = swm.World.evaluate(
            world,
            dataset=dataset,
            episodes_idx=[0],
            start_steps=[7],
        )

    assert result == {"success_rate": 100.0}
    assert world.policy is inner
    assert isinstance(inner.process["action"], _Scaler)
