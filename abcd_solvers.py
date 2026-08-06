"""Guarded A: low-budget local verification around an INTACT Direct plan."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.solver.utils import prepare_init_action


def configure_math_sdpa() -> None:
    """Use the deterministic attention backend of the paper evaluation."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


class GuardedCEMSolver(CEMSolver):
    """Preserve and locally verify a Direct plan with factorized raw-action CEM."""

    evaluation_mode = "guarded_a"

    def __init__(
        self,
        *args: Any,
        update_alpha: float = 1.0,
        std_floor: float = 1e-4,
        std_cap: float = 10.0,
        actor_covariance: bool = False,
        actor_cov_target_rms: float = 0.25,
        actor_cov_min: float = 0.05,
        actor_cov_max: float = 0.5,
        trust_lambda: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < update_alpha <= 1.0:
            raise ValueError("update_alpha must be in (0, 1]")
        if not 0.0 < std_floor <= std_cap:
            raise ValueError("require 0 < std_floor <= std_cap")
        if actor_covariance:
            raise ValueError("Canonical Guarded A fixes actor_covariance=false")
        if trust_lambda != 0.0:
            raise ValueError("Canonical Guarded A fixes trust_lambda=0")
        self.update_alpha = float(update_alpha)
        self.std_floor = float(std_floor)
        self.std_cap = float(std_cap)
        self.actor_covariance = bool(actor_covariance)
        self.actor_cov_target_rms = float(actor_cov_target_rms)
        self.actor_cov_min = float(actor_cov_min)
        self.actor_cov_max = float(actor_cov_max)
        self.trust_lambda = float(trust_lambda)
        self.timing_history: list[dict[str, float]] = []

    @staticmethod
    def _slice_info(info: dict[str, Any], start: int, end: int) -> dict[str, Any]:
        return {key: value[start:end] for key, value in info.items()}

    @staticmethod
    def _expand_info(
        info: dict[str, Any], samples: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Any]:
        expanded = {}
        for key, value in info.items():
            if torch.is_tensor(value):
                target_dtype = dtype if value.is_floating_point() else None
                value = value.to(device=device, dtype=target_dtype)
                expanded[key] = value.unsqueeze(1).expand(
                    value.shape[0], samples, *value.shape[1:]
                )
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(value[:, None, ...], samples, axis=1)
            else:
                expanded[key] = value
        return expanded

    def _score(self, info: dict[str, Any], candidates: torch.Tensor) -> torch.Tensor:
        costs = self.model.get_cost(
            self._expand_info(info, candidates.shape[1], self.device, self.dtype),
            candidates,
        ).float()
        if tuple(costs.shape) != tuple(candidates.shape[:2]):
            raise ValueError(
                f"cost shape mismatch: expected {tuple(candidates.shape[:2])}, "
                f"got {tuple(costs.shape)}"
            )
        if not torch.isfinite(costs).all():
            raise FloatingPointError("Guarded A produced non-finite costs")
        return costs

    @staticmethod
    def _gather(candidates: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(candidates.shape[0], device=candidates.device)
        return candidates[batch, index]

    @torch.inference_mode()
    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        configure_math_sdpa()
        if not bool(getattr(self.model, "has_intent_actor", lambda: False)()):
            raise RuntimeError("Guarded A requires a trained intent actor")
        if not bool(getattr(self.model, "actor_warmstart", False)):
            raise RuntimeError("Guarded A requires actor_warmstart=true")

        started = time.perf_counter()
        total_envs = len(next(iter(info_dict.values())))
        reference = prepare_init_action(
            self.model,
            info_dict,
            init_action,
            self.horizon,
            n_envs=total_envs,
            action_dim=self.action_dim,
        ).to(device=self.device, dtype=self.dtype)
        output_actions = torch.empty_like(reference)
        output_costs = torch.empty(total_envs, device=self.device, dtype=torch.float32)
        stats = defaultdict(float)
        stats.update(
            {
                "configured_num_samples": float(self.num_samples),
                "configured_iterations": float(self.n_steps),
                "configured_topk": float(self.topk),
                "configured_horizon": float(self.horizon),
                "configured_initial_std": float(self.var_scale),
                "configured_update_alpha": self.update_alpha,
                "configured_std_floor": self.std_floor,
                "configured_std_cap": self.std_cap,
                "configured_actor_covariance": float(self.actor_covariance),
                "configured_actor_cov_target_rms": self.actor_cov_target_rms,
                "configured_actor_cov_min": self.actor_cov_min,
                "configured_actor_cov_max": self.actor_cov_max,
                "configured_trust_lambda": self.trust_lambda,
                "configured_rollout_budget": float(
                    self.num_samples * self.n_steps * self.horizon
                ),
                "reference_forced_every_round": 1.0,
                "global_best_preserved": 1.0,
                "final_mean_rescored": 1.0,
                "returns_observed_choice": 1.0,
                "sdpa_math_only": 1.0,
            }
        )

        for start in range(0, total_envs, self.batch_size):
            end = min(start + self.batch_size, total_envs)
            batch_size = end - start
            info = self._slice_info(info_dict, start, end)
            batch_reference = reference[start:end]
            batch_mean = batch_reference.clone()
            batch_std = torch.full_like(batch_reference, float(self.var_scale))
            global_best = batch_reference.clone()
            global_best_cost = torch.full(
                (batch_size,), float("inf"), device=self.device
            )
            reference_cost = None

            for _ in range(self.n_steps):
                candidates = torch.randn(
                    batch_size,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                candidates = candidates * batch_std[:, None] + batch_mean[:, None]
                candidates[:, 0] = batch_reference
                if self.num_samples > 1:
                    candidates[:, 1] = global_best

                costs = self._score(info, candidates)
                stats["get_cost_calls"] += 1.0
                stats["candidate_action_sequences"] += float(
                    batch_size * self.num_samples
                )
                stats["candidate_action_steps"] += float(
                    batch_size * self.num_samples * self.horizon
                )
                if reference_cost is None:
                    reference_cost = costs[:, 0].clone()

                best_cost, best_index = costs.min(dim=1)
                best_candidate = self._gather(candidates, best_index)
                improved = best_cost < global_best_cost
                global_best_cost = torch.where(improved, best_cost, global_best_cost)
                global_best = torch.where(
                    improved[:, None, None], best_candidate, global_best
                )

                _, elite_index = torch.topk(
                    costs, k=self.topk, dim=1, largest=False
                )
                batch_index = torch.arange(batch_size, device=self.device)[:, None]
                elites = candidates[batch_index, elite_index]
                elite_mean = elites.mean(dim=1)
                elite_std = elites.std(dim=1, unbiased=False)
                alpha = self.update_alpha
                batch_mean = (1.0 - alpha) * batch_mean + alpha * elite_mean
                batch_std = (
                    (1.0 - alpha) * batch_std + alpha * elite_std
                ).clamp(min=self.std_floor, max=self.std_cap)

            final_cost = self._score(info, batch_mean[:, None])[:, 0]
            stats["get_cost_calls"] += 1.0
            stats["final_rescore_calls"] += 1.0
            improved = final_cost < global_best_cost
            global_best_cost = torch.where(improved, final_cost, global_best_cost)
            global_best = torch.where(
                improved[:, None, None], batch_mean, global_best
            )

            if reference_cost is None:
                raise RuntimeError("reference plan was never scored")
            stats["selected_reference"] += float(
                (global_best_cost == reference_cost).sum().item()
            )
            stats["predicted_non_degrading"] += float(
                (global_best_cost <= reference_cost + 1e-7).sum().item()
            )
            stats["solved_envs"] += float(batch_size)
            output_actions[start:end] = global_best
            output_costs[start:end] = global_best_cost

        stats["solve_time"] = time.perf_counter() - started
        stats["predicted_non_degrading_fraction"] = (
            stats["predicted_non_degrading"] / max(stats["solved_envs"], 1.0)
        )
        self.timing_history.append(dict(stats))
        actions = output_actions.detach().cpu()
        return {
            "actions": actions,
            "mean": [actions],
            "var": [torch.zeros_like(actions)],
            "costs": output_costs.detach().cpu().tolist(),
            "timing": dict(stats),
            "evaluation_mode": self.evaluation_mode,
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
        summary["num_solves"] = float(len(self.timing_history))
        return summary
