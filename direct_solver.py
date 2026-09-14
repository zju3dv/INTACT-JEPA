"""Zero-search solver for INTACT Direct control."""

from __future__ import annotations

import time

import torch
from stable_worldmodel.solver import CEMSolver


class DirectSolver(CEMSolver):
    """Use the INTACT action law directly without sampling or cost calls."""

    evaluation_mode = "direct"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timing_history: list[dict[str, float]] = []

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        del init_action
        start = time.perf_counter()
        actions = self.model.get_action(info_dict, horizon=self.horizon)
        stats = {
            "solve_time": time.perf_counter() - start,
            "get_cost_calls": 0.0,
            "candidate_action_sequences": 0.0,
            "candidate_action_steps": 0.0,
            "configured_candidate_sequences_per_solve": 0.0,
            "configured_candidate_action_steps_per_solve": 0.0,
            "final_mean_rescored_sequences": 0.0,
        }
        diagnostics = getattr(self.model, "last_direct_diagnostics", {})
        stats.update(
            {
                key: float(value)
                for key, value in diagnostics.items()
                if isinstance(value, (int, float))
            }
        )
        self.timing_history.append(stats)
        actions = actions.detach().cpu()
        return {
            "actions": actions,
            "mean": [actions],
            "var": [torch.zeros_like(actions)],
            "costs": [],
            "timing": stats,
            "evaluation_mode": self.evaluation_mode,
        }

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
