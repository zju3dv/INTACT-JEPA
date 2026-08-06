"""JEPA Implementation"""

import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def inverse_condition_target(current, goal, remaining, source):
    """Build either an inference waypoint or a raw terminal-goal condition."""
    if source == "waypoint":
        return current + (goal - current) / float(remaining)
    if source == "goal":
        return goal
    raise ValueError(f"Unknown inverse condition source: {source}")

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        inverse_actor=None,
        predict_residual: bool = False,
        actor_warmstart: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.inverse_actor = inverse_actor
        self.predict_residual = predict_residual
        self.actor_warmstart = actor_warmstart

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict_delta(self, emb, act_emb):
        """Predict the raw next-state output before optional residual addition.

        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def predict(self, emb, act_emb):
        """Predict next state embedding.

        With ``predict_residual=True``, the predictor output is interpreted as
        a latent delta and added to the matching source embedding.
        """
        preds = self.predict_delta(emb, act_emb)
        return emb + preds if self.predict_residual else preds

    def has_inverse_actor(self):
        return self.inverse_actor is not None

    def set_actor_warmstart(self, enabled: bool):
        self.actor_warmstart = bool(enabled)

    def get_action_dim(self, fallback=None):
        if self.inverse_actor is not None:
            return self.inverse_actor.action_dim
        if hasattr(self.action_encoder, "patch_embed"):
            return self.action_encoder.patch_embed.in_channels
        if fallback is not None:
            return fallback
        raise RuntimeError("Could not infer action dimension.")

    def inverse_action_distribution(self, z, z_next, prev_act_emb):
        """Return Gaussian action parameters for a desired latent transition."""
        if self.inverse_actor is None:
            raise RuntimeError("inverse_actor is not configured.")
        return self.inverse_actor(z, z_next, prev_act_emb)

    def inverse_action_parameters(self, z, z_next, prev_act_emb, relation=None):
        """Return a normalized dictionary of inverse-distribution parameters."""
        if self.inverse_actor is None:
            raise RuntimeError("inverse_actor is not configured.")
        if hasattr(self.inverse_actor, "distribution"):
            if relation is None:
                return self.inverse_actor.distribution(z, z_next, prev_act_emb)
            return self.inverse_actor.distribution(
                z, z_next, prev_act_emb, relation=relation
            )
        if relation is not None:
            raise RuntimeError(
                "The configured inverse actor does not support a fixed relation signal."
            )
        mean, log_std = self.inverse_actor(z, z_next, prev_act_emb)
        return {
            "kind": "gaussian",
            "mean": mean,
            "log_std": log_std,
            "map_mean": mean,
            "expectation_mean": mean,
        }

    def inverse_action_loss(
        self, z, z_next, prev_action, target_action, relation=None
    ):
        """Inverse NLL and calibration metrics for Gaussian or mixture actors."""
        prev_act_emb = self.action_encoder(prev_action)
        stats = self.inverse_action_parameters(
            z, z_next, prev_act_emb, relation=relation
        )

        if stats["kind"] == "mixture_gaussian":
            means = stats["means"]
            log_stds = stats["log_stds"]
            target = target_action.unsqueeze(-2)
            component_nll = 0.5 * (
                (target - means).square() * torch.exp(-2.0 * log_stds)
                + 2.0 * log_stds
            ).sum(dim=-1)
            mixture_log_prob = torch.logsumexp(
                stats["log_weights"] - component_nll, dim=-1
            )
            nll = -mixture_log_prob.mean() / float(self.get_action_dim())

            weights = stats["weights"]
            map_mean = stats["map_mean"]
            expectation_mean = stats["expectation_mean"]
            second_moment = (
                weights.unsqueeze(-1)
                * (torch.exp(2.0 * log_stds) + means.square())
            ).sum(dim=-2)
            mixture_var = (second_moment - expectation_mean.square()).clamp_min(1e-8)
            calibration_std = mixture_var.sqrt()

            flat_weights = weights.reshape(-1, weights.shape[-1])
            usage = flat_weights.mean(dim=0)
            hard_usage = F.one_hot(
                flat_weights.argmax(dim=-1), num_classes=weights.shape[-1]
            ).float().mean(dim=0)
            entropy_per_item = -(weights * stats["log_weights"]).sum(dim=-1)
            entropy = entropy_per_item.mean()
            effective_components = entropy_per_item.exp().mean()
            soft_collapse = (
                (usage.min() < 0.01) | (effective_components < 1.2)
            ).float()
            hard_collapse = (
                (hard_usage.min() < 0.005) | (hard_usage.max() > 0.98)
            ).float()
            collapse = torch.maximum(soft_collapse, hard_collapse)

            output = {
                "inv_loss": nll,
                "inv_mae": (map_mean - target_action).abs().mean(),
                "inv_rmse": (map_mean - target_action).square().mean().sqrt(),
                "inv_expectation_mae": (
                    expectation_mean - target_action
                ).abs().mean(),
                "inv_expectation_rmse": (
                    expectation_mean - target_action
                ).square().mean().sqrt(),
                "inv_std_mean": calibration_std.mean(),
                "inv_log_std_min": log_stds.detach().amin(),
                "inv_log_std_max": log_stds.detach().amax(),
                "inv_coverage_1sigma": (
                    (target_action - expectation_mean).abs() <= calibration_std
                ).float().mean(),
                "inv_coverage_2sigma": (
                    (target_action - expectation_mean).abs()
                    <= 2.0 * calibration_std
                ).float().mean(),
                "inv_mixture_entropy": entropy,
                "inv_effective_components": effective_components,
                "inv_soft_component_collapse": soft_collapse,
                "inv_hard_component_collapse": hard_collapse,
                "inv_component_collapse": collapse,
                "inv_mean": map_mean,
                "inv_map_mean": map_mean,
                "inv_expectation_mean": expectation_mean,
                "inv_log_std": stats["map_log_std"],
                "inv_mixture_weights": weights,
            }
            for index in range(weights.shape[-1]):
                output[f"inv_component_usage_{index}"] = usage[index]
                output[f"inv_component_hard_usage_{index}"] = hard_usage[index]
            return output

        mean = stats["mean"]
        log_std = stats["log_std"]
        std = log_std.exp()
        inv_var = torch.exp(-2.0 * log_std)
        nll = 0.5 * (
            (target_action - mean).square() * inv_var + 2.0 * log_std
        )
        return {
            "inv_loss": nll.mean(),
            "inv_mae": (mean - target_action).abs().mean(),
            "inv_rmse": (mean - target_action).square().mean().sqrt(),
            "inv_std_mean": std.mean(),
            "inv_log_std_min": log_std.detach().amin(),
            "inv_log_std_max": log_std.detach().amax(),
            "inv_coverage_1sigma": (
                (target_action - mean).abs() <= std
            ).float().mean(),
            "inv_coverage_2sigma": (
                (target_action - mean).abs() <= 2.0 * std
            ).float().mean(),
            "inv_mean": mean,
            "inv_map_mean": mean,
            "inv_expectation_mean": mean,
            "inv_log_std": log_std,
        }

    def conditional_action_loss(
        self, z, condition, relation, prev_action, target_action
    ):
        """Unified Gaussian action NLL used by every factorial cell.

        The four cells differ only in ``condition`` and ``relation``. They all
        traverse this method and the same inverse actor with identical tensor
        shapes and parameterization.
        """

        result = self.inverse_action_loss(
            z=z,
            z_next=condition,
            relation=relation,
            prev_action=prev_action,
            target_action=target_action,
        )
        return {
            "conditional_action_loss": result["inv_loss"],
            "conditional_action_mae": result["inv_mae"],
            "conditional_action_rmse": result["inv_rmse"],
            "conditional_action_std_mean": result["inv_std_mean"],
            "conditional_action_mean": result["inv_map_mean"],
            "conditional_action_log_std": result["inv_log_std"],
        }

    def goal_action_parameters(
        self, z, z_goal, prev_act_emb, steps_remaining, max_horizon
    ):
        """Action parameters for the controlled end-to-end GC-IDM objective."""
        if self.inverse_actor is None or not hasattr(
            self.inverse_actor, "goal_distribution"
        ):
            raise RuntimeError(
                "The configured inverse actor does not support goal conditioning."
            )
        return self.inverse_actor.goal_distribution(
            z,
            z_goal,
            prev_act_emb,
            steps_remaining=steps_remaining,
            max_horizon=max_horizon,
        )

    def goal_action_loss(
        self,
        z,
        z_goal,
        steps_remaining,
        prev_action,
        target_action,
        max_horizon,
    ):
        prev_act_emb = self.action_encoder(prev_action)
        stats = self.goal_action_parameters(
            z,
            z_goal,
            prev_act_emb,
            steps_remaining=steps_remaining,
            max_horizon=max_horizon,
        )
        mean = stats["mean"]
        log_std = stats["log_std"]
        std = log_std.exp()
        nll = 0.5 * (
            (target_action - mean).square() * torch.exp(-2.0 * log_std)
            + 2.0 * log_std
        )
        return {
            "goal_inv_loss": nll.mean(),
            "goal_inv_mae": (mean - target_action).abs().mean(),
            "goal_inv_rmse": (mean - target_action).square().mean().sqrt(),
            "goal_inv_std_mean": std.mean(),
            "goal_inv_coverage_1sigma": (
                (target_action - mean).abs() <= std
            ).float().mean(),
            "goal_inv_coverage_2sigma": (
                (target_action - mean).abs() <= 2.0 * std
            ).float().mean(),
            "goal_inv_mean": mean,
            "goal_inv_log_std": log_std,
        }

    def rollout_one_step(self, emb, action, act_hist, history_size: int = 3):
        """Append one latent predicted from ``action`` and return updated history.

        The newest candidate action replaces the last action in the context for
        the next prediction, matching the action-conditioned transition
        semantics used by LeWM rollout.
        """
        hs = min(history_size, emb.size(1), act_hist.size(1))
        emb_ctx = emb[:, -hs:]
        act_ctx = act_hist[:, -hs:].clone()
        act_ctx[:, -1] = action
        act_emb = self.action_encoder(act_ctx)
        pred_emb = self.predict(emb_ctx, act_emb)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)
        act_hist = torch.cat([act_hist, action[:, None]], dim=1)
        return emb, act_hist

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def _coerce_action_history(
        self,
        action: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert env single-step actions to model action-chunk vectors."""
        if action is None:
            return torch.zeros(
                batch_size, seq_len, action_dim, device=device, dtype=dtype
            )

        action = action.to(device=device, dtype=dtype)
        if action.ndim == 2:
            action = action[:, None]
        if action.ndim != 3:
            raise ValueError(
                f"Expected action history with shape (B,T,D) or (B,D), got {tuple(action.shape)}"
            )

        action = torch.nan_to_num(action.float(), 0.0).to(dtype=dtype)
        if action.shape[-1] == action_dim:
            return action
        if action_dim % action.shape[-1] == 0:
            repeats = action_dim // action.shape[-1]
            return action.repeat(*([1] * (action.ndim - 1)), repeats)
        raise ValueError(
            f"Cannot coerce action history dim {action.shape[-1]} to model action_dim {action_dim}"
        )

    @staticmethod
    def _stratified_component_indices(
        weights: torch.Tensor,
        num_samples: int,
        threshold: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Allocate a fixed population while covering every effective component."""
        weights = weights.float().clamp_min(0.0)
        weights = weights / weights.sum().clamp_min(1e-12)
        map_component = int(weights.argmax().item())
        active = torch.nonzero(weights >= threshold, as_tuple=False).flatten().tolist()
        if map_component not in active:
            active.append(map_component)
        active = sorted(active, key=lambda index: float(weights[index]), reverse=True)
        active = active[:num_samples]

        counts = torch.zeros_like(weights, dtype=torch.long)
        for index in active:
            counts[index] = 1
        remaining = num_samples - int(counts.sum().item())
        if remaining > 0:
            raw = weights * remaining
            extra = raw.floor().long()
            counts += extra
            leftover = num_samples - int(counts.sum().item())
            if leftover > 0:
                order = torch.argsort(raw - extra.float(), descending=True)
                for index in order[:leftover].tolist():
                    counts[index] += 1

        indices = torch.repeat_interleave(
            torch.arange(weights.numel(), device=weights.device), counts
        )
        if indices.numel() != num_samples:
            raise RuntimeError(
                f"component allocation produced {indices.numel()} samples, expected {num_samples}"
            )
        permutation = torch.randperm(
            num_samples, device=weights.device, generator=generator
        )
        indices = indices[permutation]
        map_positions = torch.nonzero(
            indices == map_component, as_tuple=False
        ).flatten()
        if map_positions.numel() > 0 and int(map_positions[0]) != 0:
            swap_index = int(map_positions[0])
            first = indices[0].clone()
            indices[0] = indices[swap_index]
            indices[swap_index] = first
        return indices

    @torch.inference_mode()
    def get_action_population(
        self,
        info: dict,
        horizon: int,
        num_samples: int,
        prefix_actions: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | None:
        """Sample a fixed-size autoregressive population from a MoG inverse actor."""
        if (
            not self.actor_warmstart
            or self.inverse_actor is None
            or not getattr(self.inverse_actor, "is_mixture", False)
        ):
            return None
        if horizon < 1 or num_samples < 1:
            raise ValueError(
                f"horizon and num_samples must be positive, got {horizon}, {num_samples}"
            )

        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        info = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in info.items()
        }
        batch_size = info["pixels"].size(0)
        fallback_dim = (
            prefix_actions.shape[-1]
            if prefix_actions is not None
            else info.get("action", torch.empty(0)).shape[-1]
            if torch.is_tensor(info.get("action"))
            else None
        )
        action_dim = self.get_action_dim(fallback=fallback_dim)

        init = {key: value for key, value in info.items() if torch.is_tensor(value)}
        init["action"] = self._coerce_action_history(
            init.get("action"),
            batch_size=batch_size,
            seq_len=info["pixels"].size(1),
            action_dim=action_dim,
            device=device,
            dtype=model_dtype,
        )
        init = self.encode(init)
        emb = init["emb"]
        act_hist = init["action"]
        history_size = self.predictor.pos_embedding.size(1)

        goal = {key: value for key, value in info.items() if torch.is_tensor(value)}
        goal["pixels"] = goal["goal"]
        for key in list(goal.keys()):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action", None)
        goal_emb = self.encode(goal)["emb"][:, -1]

        if prefix_actions is not None and prefix_actions.numel() > 0:
            prefix_actions = self._coerce_action_history(
                prefix_actions,
                batch_size=batch_size,
                seq_len=prefix_actions.size(1),
                action_dim=action_dim,
                device=device,
                dtype=act_hist.dtype,
            )
            for step in range(prefix_actions.size(1)):
                emb, act_hist = self.rollout_one_step(
                    emb,
                    prefix_actions[:, step],
                    act_hist,
                    history_size=history_size,
                )

        emb = (
            emb[:, None]
            .expand(batch_size, num_samples, *emb.shape[1:])
            .reshape(batch_size * num_samples, *emb.shape[1:])
            .clone()
        )
        act_hist = (
            act_hist[:, None]
            .expand(batch_size, num_samples, *act_hist.shape[1:])
            .reshape(batch_size * num_samples, *act_hist.shape[1:])
            .clone()
        )
        goal_flat = (
            goal_emb[:, None]
            .expand(batch_size, num_samples, goal_emb.shape[-1])
            .reshape(batch_size * num_samples, goal_emb.shape[-1])
        )

        actions = []
        threshold = float(
            getattr(self.inverse_actor, "effective_component_threshold", 0.05)
        )
        for step in range(horizon):
            z = emb[:, -1]
            z_next_star = z + (goal_flat - z) / float(horizon - step)
            prev_act_emb = self.action_encoder(act_hist[:, -1:])[:, -1]
            stats = self.inverse_action_parameters(z, z_next_star, prev_act_emb)
            weights = stats["weights"]

            if step == 0:
                reshaped_weights = weights.reshape(
                    batch_size, num_samples, weights.shape[-1]
                )
                components = torch.stack(
                    [
                        self._stratified_component_indices(
                            reshaped_weights[batch_index, 0],
                            num_samples,
                            threshold,
                            generator,
                        )
                        for batch_index in range(batch_size)
                    ],
                    dim=0,
                ).reshape(-1)
            else:
                components = torch.multinomial(
                    weights,
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).squeeze(-1)
                reshaped_components = components.reshape(batch_size, num_samples)
                reshaped_components[:, 0] = weights.reshape(
                    batch_size, num_samples, -1
                )[:, 0].argmax(dim=-1)
                components = reshaped_components.reshape(-1)

            gather_index = components[:, None, None].expand(
                components.shape[0], 1, action_dim
            )
            selected_mean = stats["means"].gather(-2, gather_index).squeeze(-2)
            selected_log_std = stats["log_stds"].gather(
                -2, gather_index
            ).squeeze(-2)
            noise = torch.randn(
                selected_mean.shape,
                device=selected_mean.device,
                dtype=selected_mean.dtype,
                generator=generator,
            )
            noise = noise.reshape(batch_size, num_samples, action_dim)
            noise[:, 0] = 0.0
            noise = noise.reshape(batch_size * num_samples, action_dim)
            action = selected_mean + selected_log_std.exp() * noise
            actions.append(action.reshape(batch_size, num_samples, action_dim))
            emb, act_hist = self.rollout_one_step(
                emb, action, act_hist, history_size=history_size
            )

        return torch.stack(actions, dim=2)

    @torch.inference_mode()
    def get_action(
        self,
        info: dict,
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate an inverse-actor action sequence for CEM warm-start.

        If the inverse actor is absent or disabled, this returns zero actions.
        This keeps original LeWM checkpoints compatible with the Actionable
        protocol while preserving zero-initialized CEM behavior.
        """
        assert "pixels" in info, "pixels not in info_dict"
        assert horizon >= 1, f"horizon must be positive, got {horizon}"

        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        info = {
            k: v.to(device)
            if torch.is_tensor(v)
            else v
            for k, v in info.items()
        }

        batch_size = info["pixels"].size(0)
        fallback_dim = None
        if prefix_actions is not None:
            fallback_dim = prefix_actions.shape[-1]
        elif "action" in info and torch.is_tensor(info["action"]):
            fallback_dim = info["action"].shape[-1]
        action_dim = self.get_action_dim(fallback=fallback_dim)

        if (not self.actor_warmstart) or self.inverse_actor is None:
            return torch.zeros(
                batch_size,
                horizon,
                action_dim,
                device=device,
                dtype=model_dtype,
            )

        init = {k: v for k, v in info.items() if torch.is_tensor(v)}
        init["action"] = self._coerce_action_history(
            init.get("action"),
            batch_size=batch_size,
            seq_len=info["pixels"].size(1),
            action_dim=action_dim,
            device=device,
            dtype=model_dtype,
        )
        init = self.encode(init)

        emb = init["emb"]
        act_hist = init["action"]
        history_size = self.predictor.pos_embedding.size(1)

        goal = {k: v for k, v in info.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in list(goal.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)
        goal.pop("action", None)
        goal = self.encode(goal)
        goal_emb = goal["emb"][:, -1]

        if prefix_actions is not None and prefix_actions.numel() > 0:
            prefix_actions = self._coerce_action_history(
                prefix_actions,
                batch_size=batch_size,
                seq_len=prefix_actions.size(1),
                action_dim=action_dim,
                device=device,
                dtype=act_hist.dtype,
            )
            for t in range(prefix_actions.size(1)):
                emb, act_hist = self.rollout_one_step(
                    emb, prefix_actions[:, t], act_hist, history_size=history_size
                )

        actions = []
        direct_target_mode = os.environ.get(
            "INVERSE_DIRECT_TARGET_MODE", "query"
        ).lower()
        if direct_target_mode not in {
            "query",
            "query_horizon",
            "goal",
            "goal_no_horizon",
        }:
            raise ValueError(
                "INVERSE_DIRECT_TARGET_MODE must be query, query_horizon, "
                "goal, or goal_no_horizon."
            )
        bypass_predictor = os.environ.get(
            "INVERSE_QUERY_BYPASS_PREDICTOR", "0"
        ).lower() in {"1", "true", "yes"}
        beta_mode = os.environ.get("INVERSE_QUERY_BETA_MODE", "off").lower()
        if beta_mode not in {"off", "progress", "predictor"}:
            raise ValueError(
                "INVERSE_QUERY_BETA_MODE must be off, progress, or predictor."
            )
        if beta_mode == "predictor" and bypass_predictor:
            raise ValueError(
                "predictor Beta inference is incompatible with predictor bypass."
            )
        beta_k = int(os.environ.get("INVERSE_QUERY_BETA_K", "1"))
        if beta_k < 1:
            raise ValueError("INVERSE_QUERY_BETA_K must be positive.")
        beta_fixed_text = os.environ.get("INVERSE_QUERY_BETA_FIXED", "").strip()
        beta_fixed = float(beta_fixed_text) if beta_fixed_text else None
        if beta_fixed is not None and not 0.0 <= beta_fixed <= 1.0:
            raise ValueError("INVERSE_QUERY_BETA_FIXED must be in [0, 1].")
        beta_alpha = float(os.environ.get("INVERSE_QUERY_BETA_ALPHA", "0.5"))
        if beta_alpha <= 0.0:
            raise ValueError("INVERSE_QUERY_BETA_ALPHA must be positive.")

        def inverse_mean(z, target, previous_action_embedding, relation=None):
            if relation is not None:
                return self.inverse_action_parameters(
                    z,
                    target,
                    previous_action_embedding,
                    relation=relation,
                )["map_mean"]
            if hasattr(self.inverse_actor, "action_mean"):
                return self.inverse_actor.action_mean(
                    z, target, previous_action_embedding
                )
            mean, _ = self.inverse_action_distribution(
                z, target, previous_action_embedding
            )
            return mean

        def beta_fused_mean(z, anchor, target, previous_action_embedding):
            if beta_fixed is not None:
                beta = z.new_full((z.size(0), beta_k, 1), beta_fixed)
            else:
                concentration = torch.tensor(
                    beta_alpha, device=z.device, dtype=torch.float32
                )
                beta = torch.distributions.Beta(
                    concentration, concentration
                ).sample((z.size(0), beta_k, 1)).to(dtype=z.dtype)

            fused_targets = anchor[:, None] + beta * (
                target - anchor
            )[:, None]
            z_flat = z[:, None].expand_as(fused_targets).reshape(
                z.size(0) * beta_k, z.size(-1)
            )
            target_flat = fused_targets.reshape(
                z.size(0) * beta_k, z.size(-1)
            )
            previous_flat = previous_action_embedding[:, None].expand(
                z.size(0), beta_k, previous_action_embedding.size(-1)
            ).reshape(z.size(0) * beta_k, previous_action_embedding.size(-1))
            means = inverse_mean(z_flat, target_flat, previous_flat)
            return means.reshape(z.size(0), beta_k, -1).mean(dim=1)

        condition_source = os.environ.get(
            "INVERSE_QUERY_CONDITION_SOURCE", "goal"
        ).lower()
        if condition_source not in {"waypoint", "goal"}:
            raise ValueError(
                "INVERSE_QUERY_CONDITION_SOURCE must be waypoint or goal."
            )
        for t in range(horizon):
            z = emb[:, -1]
            gamma = float(os.environ.get("INVERSE_QUERY_GAMMA", "1.0"))
            scale = float(os.environ.get("INVERSE_QUERY_SCALE", "1.0"))
            if gamma <= 0.0 or scale <= 0.0:
                raise ValueError("INVERSE_QUERY_GAMMA and INVERSE_QUERY_SCALE must be positive.")
            progress_now = (float(t) / float(horizon)) ** gamma
            progress_next = (float(t + 1) / float(horizon)) ** gamma
            fraction = scale * (progress_next - progress_now) / max(
                1.0 - progress_now, 1e-8
            )
            fraction = min(fraction, 1.0)
            z_next_star = inverse_condition_target(
                z, goal_emb, horizon - t, condition_source
            )
            prev_act_emb = self.action_encoder(act_hist[:, -1:])[:, -1]
            if direct_target_mode in {"goal", "goal_no_horizon"}:
                steps_remaining = torch.full(
                    (z.size(0),),
                    horizon - t,
                    device=z.device,
                    dtype=torch.long,
                )
                relation = (
                    self.inverse_actor.horizon_embedding(
                        steps_remaining,
                        z,
                        max_horizon=horizon,
                    )
                    if direct_target_mode == "goal"
                    else torch.zeros_like(z)
                )
                mean = inverse_mean(
                    z, goal_emb, prev_act_emb, relation=relation
                )
            elif direct_target_mode == "query_horizon":
                steps_remaining = torch.full(
                    (z.size(0),),
                    horizon - t,
                    device=z.device,
                    dtype=torch.long,
                )
                relation = self.inverse_actor.horizon_embedding(
                    steps_remaining,
                    z,
                    max_horizon=horizon,
                )
                mean = inverse_mean(
                    z, z_next_star, prev_act_emb, relation=relation
                )
            elif beta_mode == "off":
                mean = inverse_mean(z, z_next_star, prev_act_emb)
            elif beta_mode == "progress":
                mean = beta_fused_mean(
                    z, z, z_next_star, prev_act_emb
                )
            else:
                base_mean = inverse_mean(z, z_next_star, prev_act_emb)
                anchor_emb, _ = self.rollout_one_step(
                    emb, base_mean, act_hist, history_size=history_size
                )
                mean = beta_fused_mean(
                    z, anchor_emb[:, -1], z_next_star, prev_act_emb
                )
            actions.append(mean)
            if bypass_predictor:
                emb = torch.cat([emb, z_next_star[:, None]], dim=1)
                act_hist = torch.cat([act_hist, mean[:, None]], dim=1)
            else:
                emb, act_hist = self.rollout_one_step(
                    emb, mean, act_hist, history_size=history_size
                )

        return torch.stack(actions, dim=1)

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
