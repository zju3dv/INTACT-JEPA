"""INTACT: a LeWM-style JEPA with a shared intent-to-action operator."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class JEPA(nn.Module):
    """Forward world model plus a deployable local/goal intent action law."""

    VALID_INTENT_MODES = {"waypoint", "goal_displacement"}

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        intent_actor: nn.Module | None = None,
        intent_mode: str = "goal_displacement",
        predict_residual: bool = False,
        actor_warmstart: bool = True,
    ) -> None:
        super().__init__()
        if intent_mode not in self.VALID_INTENT_MODES:
            raise ValueError(
                f"intent_mode must be one of {sorted(self.VALID_INTENT_MODES)}, got {intent_mode}"
            )
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.intent_actor = intent_actor
        self.intent_mode = intent_mode
        self.predict_residual = predict_residual
        self.actor_warmstart = actor_warmstart

    def encode(self, info: dict) -> dict:
        pixels = info["pixels"].float()
        batch_size = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        encoded = self.encoder(pixels, interpolate_pos_encoding=True)
        embedding = self.projector(encoded.last_hidden_state[:, 0])
        info["emb"] = rearrange(
            embedding, "(b t) d -> b t d", b=batch_size
        )
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, embedding: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        prediction = self.predictor(embedding, action_embedding)
        prediction = self.pred_proj(rearrange(prediction, "b t d -> (b t) d"))
        prediction = rearrange(
            prediction, "(b t) d -> b t d", b=embedding.size(0)
        )
        return embedding + prediction if self.predict_residual else prediction

    def has_intent_actor(self) -> bool:
        return self.intent_actor is not None

    def set_actor_warmstart(self, enabled: bool) -> None:
        self.actor_warmstart = bool(enabled)

    def get_action_dim(self, fallback: int | None = None) -> int:
        if self.intent_actor is not None:
            return self.intent_actor.action_dim
        if hasattr(self.action_encoder, "patch_embed"):
            return self.action_encoder.patch_embed.in_channels
        if fallback is not None:
            return fallback
        raise RuntimeError("Could not infer action dimension")

    def goal_intent(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        steps_remaining: int | float,
    ) -> torch.Tensor:
        displacement = goal - current
        if self.intent_mode == "waypoint":
            if float(steps_remaining) <= 0:
                raise ValueError("steps_remaining must be positive")
            return displacement / float(steps_remaining)
        return displacement

    def action_parameters(
        self,
        z: torch.Tensor,
        intent: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.intent_actor is None:
            raise RuntimeError("intent_actor is not configured")
        previous_embedding = self.action_encoder(previous_action)
        return self.intent_actor(z, intent, previous_embedding)

    def action_nll(
        self,
        z: torch.Tensor,
        intent: torch.Tensor,
        previous_action: torch.Tensor,
        target_action: torch.Tensor,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        mean, log_std = self.action_parameters(z, intent, previous_action)
        nll = 0.5 * (
            (target_action - mean).square() * torch.exp(-2 * log_std)
            + 2 * log_std
        ).mean(dim=-1)
        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "none":
            loss = nll
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
        return {
            "loss": loss,
            "nll": nll,
            "mean": mean,
            "log_std": log_std,
            "mae": (mean - target_action).abs().mean(dim=-1),
            "rmse": (mean - target_action).square().mean(dim=-1).sqrt(),
        }

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
        if action is None:
            raise ValueError(
                "Action history is required; initialize reset history as raw "
                "zero before applying the action normalizer"
            )
        action = action.to(device=device, dtype=dtype)
        if not torch.isfinite(action).all():
            raise ValueError(
                "Action history contains non-finite values; initialize missing "
                "history as raw zero before action normalization"
            )
        if action.ndim == 2:
            action = action[:, None]
        if action.ndim != 3:
            raise ValueError(f"Expected action history [B,T,D], got {action.shape}")
        if action.size(-1) == action_dim:
            return action
        if action_dim % action.size(-1) == 0:
            return action.repeat(1, 1, action_dim // action.size(-1))
        raise ValueError(
            f"Cannot coerce action dimension {action.size(-1)} to {action_dim}"
        )

    def rollout_one_step(
        self,
        embedding_history: torch.Tensor,
        action: torch.Tensor,
        action_history: torch.Tensor,
        history_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = min(history_size, embedding_history.size(1), action_history.size(1))
        embedding_context = embedding_history[:, -size:]
        action_context = action_history[:, -size:].clone()
        action_context[:, -1] = action
        prediction = self.predict(
            embedding_context, self.action_encoder(action_context)
        )[:, -1:]
        return (
            torch.cat([embedding_history, prediction], dim=1),
            torch.cat([action_history, action[:, None]], dim=1),
        )

    def _encode_goal(self, info: dict) -> torch.Tensor:
        goal = {key: value for key, value in info.items() if torch.is_tensor(value)}
        goal["pixels"] = goal["goal"]
        for key in list(goal):
            if key.startswith("goal_"):
                goal[key.removeprefix("goal_")] = goal.pop(key)
        goal.pop("action", None)
        return self.encode(goal)["emb"][:, -1]

    @torch.inference_mode()
    def get_action(
        self,
        info: dict,
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a deterministic Direct plan with zero candidate search."""
        if horizon < 1:
            raise ValueError("horizon must be positive")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        info = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in info.items()
        }
        batch_size = info["pixels"].size(0)
        fallback = None
        if prefix_actions is not None:
            fallback = prefix_actions.size(-1)
        elif torch.is_tensor(info.get("action")):
            fallback = info["action"].size(-1)
        action_dim = self.get_action_dim(fallback)

        if not self.actor_warmstart or self.intent_actor is None:
            return torch.zeros(
                batch_size, horizon, action_dim, device=device, dtype=dtype
            )

        initial = {key: value for key, value in info.items() if torch.is_tensor(value)}
        initial["action"] = self._coerce_action_history(
            initial.get("action"),
            batch_size=batch_size,
            seq_len=info["pixels"].size(1),
            action_dim=action_dim,
            device=device,
            dtype=dtype,
        )
        initial = self.encode(initial)
        embeddings = initial["emb"]
        actions_so_far = initial["action"]
        history_size = self.predictor.pos_embedding.size(1)
        goal = self._encode_goal(info)

        if prefix_actions is not None and prefix_actions.numel() > 0:
            prefix_actions = self._coerce_action_history(
                prefix_actions,
                batch_size=batch_size,
                seq_len=prefix_actions.size(1),
                action_dim=action_dim,
                device=device,
                dtype=actions_so_far.dtype,
            )
            for step in range(prefix_actions.size(1)):
                embeddings, actions_so_far = self.rollout_one_step(
                    embeddings,
                    prefix_actions[:, step],
                    actions_so_far,
                    history_size,
                )

        plan = []
        intent_norms = []
        for step in range(horizon):
            current = embeddings[:, -1]
            intent = self.goal_intent(current, goal, horizon - step)
            previous_embedding = self.action_encoder(actions_so_far[:, -1:])[:, -1]
            action = self.intent_actor.action_mean(
                current, intent, previous_embedding
            )
            plan.append(action)
            intent_norms.append(intent.norm(dim=-1).mean())
            embeddings, actions_so_far = self.rollout_one_step(
                embeddings, action, actions_so_far, history_size
            )

        self.last_direct_diagnostics = {
            "intent_norm": float(torch.stack(intent_norms).mean()),
            "terminal_latent_error": float(
                (embeddings[:, -1] - goal).norm(dim=-1).mean()
            ),
            "forward_calls": float(horizon),
            "candidate_sequences": 0.0,
        }
        return torch.stack(plan, dim=1)

    def rollout(
        self, info: dict, action_sequence: torch.Tensor, history_size: int | None = None
    ) -> dict:
        """Roll out action candidates for optional LeWM-compatible CEM checks."""
        batch_size, samples, horizon = action_sequence.shape[:3]
        observed_steps = info["pixels"].size(2)
        initial_actions, future_actions = torch.split(
            action_sequence, [observed_steps, horizon - observed_steps], dim=2
        )
        initial = {
            key: value[:, 0]
            for key, value in info.items()
            if torch.is_tensor(value)
        }
        initial["action"] = initial_actions[:, 0]
        initial = self.encode(initial)
        embeddings = initial["emb"].unsqueeze(1).expand(
            batch_size, samples, -1, -1
        )
        embeddings = rearrange(embeddings, "b s t d -> (b s) t d").clone()
        actions = rearrange(initial_actions, "b s t d -> (b s) t d")
        future_actions = rearrange(future_actions, "b s t d -> (b s) t d")
        context = history_size or self.predictor.pos_embedding.size(1)

        for step in range(horizon - observed_steps):
            prediction = self.predict(
                embeddings[:, -context:],
                self.action_encoder(actions)[:, -context:],
            )[:, -1:]
            embeddings = torch.cat([embeddings, prediction], dim=1)
            actions = torch.cat(
                [actions, future_actions[:, step : step + 1]], dim=1
            )

        final_prediction = self.predict(
            embeddings[:, -context:], self.action_encoder(actions)[:, -context:]
        )[:, -1:]
        embeddings = torch.cat([embeddings, final_prediction], dim=1)
        info["predicted_emb"] = rearrange(
            embeddings, "(b s) t d -> b s t d", b=batch_size, s=samples
        )
        return info

    def criterion(self, info: dict) -> torch.Tensor:
        predicted = info["predicted_emb"][:, :, -1]
        goal = info["goal_emb"][:, None].expand_as(predicted)
        return F.mse_loss(predicted, goal.detach(), reduction="none").sum(dim=-1)

    def get_cost(self, info: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        info = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in info.items()
        }
        collapsed = {
            key: value[:, 0]
            for key, value in info.items()
            if torch.is_tensor(value)
        }
        info["goal_emb"] = self._encode_goal(collapsed)
        return self.criterion(self.rollout(info, action_candidates))
