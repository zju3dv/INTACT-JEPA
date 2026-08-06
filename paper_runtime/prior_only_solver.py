"""Solver that executes the Inverse LeWM proposal without CEM refinement."""

from __future__ import annotations

import time

import torch
from stable_worldmodel.solver import CEMSolver


class PriorOnlySolver(CEMSolver):
    """Reuse CEM configuration plumbing while bypassing candidate search."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_history: list[dict[str, float]] = []

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        del init_action
        start = time.time()
        actions = self.model.get_action(info_dict, horizon=self.horizon)
        elapsed = time.time() - start
        stats = {
            "solve_time": elapsed,
            "get_cost_calls": 0.0,
            "candidate_action_steps": 0.0,
            "configured_rollout_budget": 0.0,
            "actor_warmstart_enabled": float(
                bool(getattr(self.model, "actor_warmstart", False))
            ),
        }
        self.timing_history.append(stats)
        actions = actions.detach().cpu()
        return {
            "actions": actions,
            "mean": [actions],
            "var": [torch.zeros_like(actions)],
            "costs": [],
            "timing": stats,
        }

    def timing_summary(self) -> dict[str, float]:
        if not self.timing_history:
            return {}
        keys = sorted({key for row in self.timing_history for key in row})
        summary = {}
        for key in keys:
            values = [float(row.get(key, 0.0)) for row in self.timing_history]
            summary[f"{key}_sum"] = sum(values)
            summary[f"{key}_mean"] = sum(values) / len(values)
        summary["num_solves"] = len(self.timing_history)
        return summary
