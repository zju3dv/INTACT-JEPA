"""Explicit CEM variants for clean INTACT evaluation."""

from __future__ import annotations

import math
import time

import torch
from stable_worldmodel.solver import CEMSolver


class _TimedCEMSolver(CEMSolver):
    evaluation_mode = "cem"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timing_history: list[dict[str, float]] = []

    def _solve_and_record(self, info_dict: dict, init_action: torch.Tensor | None) -> dict:
        started = time.perf_counter()
        outputs = super().solve(info_dict, init_action=init_action)
        num_envs = len(next(iter(info_dict.values())))
        stats = {
            "solve_time": time.perf_counter() - started,
            "get_cost_calls": float(self.n_steps * math.ceil(num_envs / self.batch_size)),
            "candidate_action_steps": float(
                num_envs * self.n_steps * self.num_samples * self.horizon
            ),
            "configured_rollout_budget": float(
                self.n_steps * self.num_samples * self.horizon
            ),
        }
        self.timing_history.append(stats)
        outputs["timing"] = stats
        outputs["evaluation_mode"] = self.evaluation_mode
        return outputs

    def timing_summary(self) -> dict[str, float]:
        if not self.timing_history:
            return {}
        keys = sorted({key for row in self.timing_history for key in row})
        summary = {}
        for key in keys:
            values = [row.get(key, 0.0) for row in self.timing_history]
            summary[f"{key}_sum"] = sum(values)
            summary[f"{key}_mean"] = sum(values) / len(values)
        summary["num_solves"] = float(len(self.timing_history))
        return summary


class PureCEMSolver(_TimedCEMSolver):
    """CEM with a zero-mean initial distribution and no actor call."""

    evaluation_mode = "pure_cem"

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        num_envs = len(next(iter(info_dict.values())))
        if init_action is None:
            init_action = torch.zeros(
                num_envs, self.horizon, self.action_dim, dtype=self.dtype
            )
        return self._solve_and_record(info_dict, init_action)


class ActorCEMSolver(_TimedCEMSolver):
    """CEM initialized by the checkpoint's intent actor plan."""

    evaluation_mode = "actor_cem"

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        if not callable(getattr(self.model, "get_action", None)):
            raise TypeError("Actor-CEM requires a model implementing get_action")
        if not bool(getattr(self.model, "has_intent_actor", lambda: False)()):
            raise RuntimeError("Actor-CEM requires a trained intent actor")
        if not bool(getattr(self.model, "actor_warmstart", False)):
            raise RuntimeError("Actor-CEM requires actor_warmstart=true")
        if init_action is not None:
            raise ValueError("Actor-CEM does not accept an external init_action")
        return self._solve_and_record(info_dict, None)
