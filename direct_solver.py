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
        if not bool(getattr(self.model, "has_intent_actor", lambda: False)()):
            raise RuntimeError("Direct evaluation requires a trained intent actor")
        start = time.perf_counter()
        actions = self.model.get_action(info_dict, horizon=self.horizon)
        stats = {
            "solve_time": time.perf_counter() - start,
            "get_cost_calls": 0.0,
            "candidate_sequences": 0.0,
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
        }

    def timing_summary(self) -> dict[str, float]:
        if not self.timing_history:
            return {}
        keys = sorted({key for row in self.timing_history for key in row})
        summary = {
            f"{key}_mean": sum(row.get(key, 0.0) for row in self.timing_history)
            / len(self.timing_history)
            for key in keys
        }
        summary["num_solves"] = float(len(self.timing_history))
        return summary
