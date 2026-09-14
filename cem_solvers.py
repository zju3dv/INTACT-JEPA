"""Explicit CEM variants for clean INTACT evaluation."""

from __future__ import annotations

import math
import time

import torch
from stable_worldmodel.solver import CEMSolver


class _TimedCEMSolver(CEMSolver):
    """CEM solver with mode identity and aggregate timing diagnostics."""

    evaluation_mode = "cem"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timing_history: list[dict[str, float]] = []

    def _solve_and_record(
        self, info_dict: dict, init_action: torch.Tensor | None
    ) -> dict:
        started = time.perf_counter()
        outputs = super().solve(info_dict, init_action=init_action)
        num_envs = len(next(iter(info_dict.values())))
        stats = {
            "solve_time": time.perf_counter() - started,
            "get_cost_calls": float(self.n_steps * math.ceil(num_envs / self.batch_size)),
            "candidate_action_sequences": float(
                num_envs * self.n_steps * self.num_samples
            ),
            "candidate_action_steps": float(
                num_envs * self.n_steps * self.num_samples * self.horizon
            ),
            "configured_candidate_sequences_per_solve": float(
                self.n_steps * self.num_samples
            ),
            "configured_candidate_action_steps_per_solve": float(
                self.n_steps * self.num_samples * self.horizon
            ),
            "final_mean_rescored_sequences": 0.0,
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
    """Run CEM from its zero-mean search distribution without Actor calls."""

    evaluation_mode = "pure_cem"

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        num_envs = len(next(iter(info_dict.values())))
        if init_action is None:
            init_action = torch.zeros(
                num_envs, self.horizon, self.action_dim, dtype=self.dtype
            )
        else:
            if init_action.ndim != 3:
                raise ValueError(
                    f"Expected init_action [B,T,D], got {init_action.shape}"
                )
            if init_action.size(0) != num_envs or init_action.size(2) != self.action_dim:
                raise ValueError(
                    "init_action shape does not match the configured environments/action space"
                )
            if init_action.size(1) < self.horizon:
                tail = init_action.new_zeros(
                    num_envs, self.horizon - init_action.size(1), self.action_dim
                )
                init_action = torch.cat([init_action, tail], dim=1)
            else:
                init_action = init_action[:, : self.horizon]
        return self._solve_and_record(info_dict, init_action)


class ActorCEMSolver(_TimedCEMSolver):
    """Run CEM with the Fig. 1 Actor plan as the initial distribution mean."""

    evaluation_mode = "actor_cem"

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        if not callable(getattr(self.model, "get_action", None)):
            raise TypeError("Actor CEM requires a model implementing get_action")
        if init_action is not None:
            raise ValueError(
                "Actor CEM constructs its initial mean from the Actor; "
                "external init_action is not supported"
            )
        action_history = info_dict.get("action")
        if not torch.is_tensor(action_history) or action_history.ndim < 2:
            raise ValueError(
                "Actor CEM requires a real action history in info_dict['action']"
            )
        if action_history.numel() == 0 or not torch.isfinite(action_history).all():
            raise ValueError("Actor CEM action history contains non-finite values")
        return self._solve_and_record(info_dict, None)
