import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class InverseTransitionActor(nn.Module):
    """Predict an action distribution from a desired latent transition.

    The actor intentionally receives more than just ``z_next - z``. Latent
    coordinates can rotate or rescale during training, so the concatenated
    state, target, difference, and product give the MLP enough local geometry
    to learn a stable inverse map.
    """

    def __init__(
        self,
        *,
        embed_dim,
        action_emb_dim,
        action_dim,
        hidden_dim=1024,
        depth=3,
        dropout=0.0,
        min_log_std=-5.0,
        max_log_std=2.0,
        use_delta=True,
        use_product=True,
        feature_layout="endpoint_delta_product",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_emb_dim = action_emb_dim
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.use_delta = bool(use_delta)
        self.use_product = bool(use_product)
        if feature_layout not in {
            "endpoint_delta_product",
            "delta_condition_product",
        }:
            raise ValueError(f"Unknown inverse actor feature layout: {feature_layout}")
        self.feature_layout = feature_layout
        self.is_mixture = False

        # Keep the first-layer shape fixed across feature ablations so that
        # parameter count and initialization remain controlled.
        input_dim = 4 * embed_dim + action_emb_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(max(depth - 1, 0)):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 2 * action_dim))
        self.net = nn.Sequential(*layers)

    def actor_features(self, z, z_next, prev_act_emb, relation=None):
        """Build the audited, fixed-width actor input tensor."""
        if self.feature_layout == "delta_condition_product":
            if relation is not None:
                raise ValueError(
                    "delta_condition_product requires an empty relation slot"
                )
            delta = z_next - z
            return torch.cat(
                [z, delta, torch.zeros_like(z), z * delta, prev_act_emb], dim=-1
            )

        if relation is None:
            relation = z_next - z if self.use_delta else torch.zeros_like(z)
        if relation.shape != z.shape:
            relation = relation.expand_as(z)
        product = z * z_next if self.use_product else torch.zeros_like(z)
        return torch.cat([z, z_next, relation, product, prev_act_emb], dim=-1)

    def _actor_parameters(self, z, z_next, prev_act_emb, relation=None):
        feat = self.actor_features(z, z_next, prev_act_emb, relation=relation)
        mean, log_std = self.net(feat).chunk(2, dim=-1)
        return mean, log_std.clamp(self.min_log_std, self.max_log_std)

    def forward(self, z, z_next, prev_act_emb):
        """
        z: (..., D)
        z_next: (..., D), desired next latent
        prev_act_emb: (..., A_emb), previous-action context only
        """
        return self._actor_parameters(z, z_next, prev_act_emb)

    def horizon_embedding(self, steps_remaining, reference, max_horizon):
        """Official 64-D GC-IDM sinusoid, zero-padded into the relation slot."""
        steps = torch.as_tensor(
            steps_remaining, device=reference.device, dtype=reference.dtype
        )
        while steps.ndim < reference.ndim - 1:
            steps = steps.unsqueeze(-1)
        fraction = steps / float(max_horizon)
        horizon_dim = min(64, self.embed_dim)
        horizon_dim -= horizon_dim % 2
        half = horizon_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=reference.device, dtype=reference.dtype)
            / float(half)
        )
        angles = fraction.unsqueeze(-1) * frequencies
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.size(-1) < self.embed_dim:
            embedding = F.pad(embedding, (0, self.embed_dim - embedding.size(-1)))
        return embedding

    def goal_distribution(
        self, z, z_goal, prev_act_emb, steps_remaining, max_horizon
    ):
        """Goal-conditioned action distribution with explicit remaining horizon."""
        horizon = self.horizon_embedding(steps_remaining, z, max_horizon)
        mean, log_std = self._actor_parameters(
            z, z_goal, prev_act_emb, relation=horizon
        )
        return {
            "kind": "gaussian",
            "mean": mean,
            "log_std": log_std,
            "map_mean": mean,
            "expectation_mean": mean,
        }

    def goal_action_mean(
        self, z, z_goal, prev_act_emb, steps_remaining, max_horizon
    ):
        return self.goal_distribution(
            z, z_goal, prev_act_emb, steps_remaining, max_horizon
        )["map_mean"]

    def distribution(self, z, z_next, prev_act_emb, relation=None):
        mean, log_std = self._actor_parameters(
            z, z_next, prev_act_emb, relation=relation
        )
        return {
            "kind": "gaussian",
            "mean": mean,
            "log_std": log_std,
            "map_mean": mean,
            "expectation_mean": mean,
        }

    def action_mean(self, z, z_next, prev_act_emb):
        mean, _ = self(z, z_next, prev_act_emb)
        return mean


def _branch_mlp(
    input_dim,
    hidden_dim,
    output_dim,
    *,
    depth=3,
    dropout=0.0,
):
    if depth < 1:
        raise ValueError(f"depth must be positive, got {depth}")
    layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
    for _ in range(depth - 1):
        layers.extend(
            [
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ]
        )
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def _branch_trunk(input_dim, hidden_dim, *, depth=3, dropout=0.0):
    if depth < 1:
        raise ValueError(f"depth must be positive, got {depth}")
    layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
    for _ in range(depth - 1):
        layers.extend(
            [
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ]
        )
    return nn.Sequential(*layers)


class MidpointBilinearSplitInverseActor(nn.Module):
    """Independent mean and scale branches over midpoint/transition geometry."""

    def __init__(
        self,
        *,
        embed_dim,
        action_emb_dim,
        action_dim,
        projection_dim=None,
        hidden_dim=704,
        depth=3,
        dropout=0.0,
        min_log_std=-5.0,
        max_log_std=2.0,
    ):
        super().__init__()
        projection_dim = projection_dim or embed_dim
        self.embed_dim = embed_dim
        self.action_emb_dim = action_emb_dim
        self.action_dim = action_dim
        self.projection_dim = projection_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.is_mixture = False

        self.u_m = nn.Linear(embed_dim, projection_dim)
        self.u_d = nn.Linear(embed_dim, projection_dim)
        branch_input_dim = 2 * embed_dim + projection_dim + action_emb_dim
        self.mu_net = _branch_mlp(
            branch_input_dim,
            hidden_dim,
            action_dim,
            depth=depth,
            dropout=dropout,
        )
        self.log_std_net = _branch_mlp(
            branch_input_dim,
            hidden_dim,
            action_dim,
            depth=depth,
            dropout=dropout,
        )

    def transition_features(self, z, z_next):
        scale = math.sqrt(2.0)
        midpoint = (z + z_next) / scale
        delta = (z_next - z) / scale
        interaction = self.u_m(midpoint) * self.u_d(delta)
        return midpoint, delta, interaction

    def forward(self, z, z_next, prev_act_emb):
        midpoint, delta, interaction = self.transition_features(z, z_next)
        mu_features = torch.cat(
            [midpoint, delta, interaction, prev_act_emb], dim=-1
        )
        scale_features = torch.cat(
            [midpoint, delta.square(), interaction.abs(), prev_act_emb], dim=-1
        )
        mean = self.mu_net(mu_features)
        log_std = self.log_std_net(scale_features).clamp(
            self.min_log_std, self.max_log_std
        )
        return mean, log_std

    def distribution(self, z, z_next, prev_act_emb):
        mean, log_std = self(z, z_next, prev_act_emb)
        return {
            "kind": "gaussian",
            "mean": mean,
            "log_std": log_std,
            "map_mean": mean,
            "expectation_mean": mean,
        }

    def action_mean(self, z, z_next, prev_act_emb):
        mean, _ = self(z, z_next, prev_act_emb)
        return mean


class ConditionalJacobianInverseActor(nn.Module):
    """Low-rank state-conditioned local inverse Jacobian."""

    def __init__(
        self,
        *,
        embed_dim,
        action_emb_dim,
        action_dim,
        rank=128,
        hidden_dim=768,
        context_depth=1,
        base_depth=2,
        scale_depth=3,
        dropout=0.0,
        min_log_std=-5.0,
        max_log_std=2.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_emb_dim = action_emb_dim
        self.action_dim = action_dim
        self.rank = rank
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.is_mixture = False

        context_input_dim = embed_dim + action_emb_dim
        self.context_net = _branch_mlp(
            context_input_dim,
            hidden_dim,
            embed_dim,
            depth=context_depth,
            dropout=dropout,
        )
        self.base_net = _branch_mlp(
            context_input_dim,
            hidden_dim,
            action_dim,
            depth=base_depth,
            dropout=dropout,
        )
        self.u_c = nn.Linear(embed_dim, rank)
        self.u_d = nn.Linear(embed_dim, rank)
        self.w_o = nn.Linear(rank, action_dim)

        scale_input_dim = 2 * embed_dim + rank + action_emb_dim
        self.log_std_net = _branch_mlp(
            scale_input_dim,
            hidden_dim,
            action_dim,
            depth=scale_depth,
            dropout=dropout,
        )

    def transition_features(self, z, z_next, prev_act_emb):
        scale = math.sqrt(2.0)
        midpoint = (z + z_next) / scale
        delta = (z_next - z) / scale
        context_input = torch.cat([midpoint, prev_act_emb], dim=-1)
        context = self.context_net(context_input)
        interaction = self.u_c(context) * self.u_d(delta)
        return midpoint, delta, context_input, interaction

    def forward(self, z, z_next, prev_act_emb):
        midpoint, delta, context_input, interaction = self.transition_features(
            z, z_next, prev_act_emb
        )
        mean = self.base_net(context_input) + self.w_o(interaction)
        scale_features = torch.cat(
            [midpoint, delta.square(), interaction.abs(), prev_act_emb], dim=-1
        )
        log_std = self.log_std_net(scale_features).clamp(
            self.min_log_std, self.max_log_std
        )
        return mean, log_std

    def distribution(self, z, z_next, prev_act_emb):
        mean, log_std = self(z, z_next, prev_act_emb)
        return {
            "kind": "gaussian",
            "mean": mean,
            "log_std": log_std,
            "map_mean": mean,
            "expectation_mean": mean,
        }

    def action_mean(self, z, z_next, prev_act_emb):
        mean, _ = self(z, z_next, prev_act_emb)
        return mean


class MixtureGaussianInverseActor(nn.Module):
    """Three-component diagonal Gaussian inverse transition distribution."""

    def __init__(
        self,
        *,
        embed_dim,
        action_emb_dim,
        action_dim,
        num_components=3,
        projection_dim=None,
        hidden_dim=1024,
        depth=3,
        dropout=0.0,
        min_log_std=-5.0,
        max_log_std=2.0,
        planning_mode="mog_map",
        effective_component_threshold=0.05,
    ):
        super().__init__()
        projection_dim = projection_dim or embed_dim
        self.embed_dim = embed_dim
        self.action_emb_dim = action_emb_dim
        self.action_dim = action_dim
        self.num_components = num_components
        self.projection_dim = projection_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.planning_mode = planning_mode
        self.effective_component_threshold = effective_component_threshold
        self.is_mixture = True

        self.u_m = nn.Linear(embed_dim, projection_dim)
        self.u_d = nn.Linear(embed_dim, projection_dim)
        input_dim = 2 * embed_dim + projection_dim + action_emb_dim
        self.trunk = _branch_trunk(
            input_dim, hidden_dim, depth=depth, dropout=dropout
        )
        self.logits_head = nn.Linear(hidden_dim, num_components)
        self.means_head = nn.Linear(
            hidden_dim, num_components * action_dim
        )
        self.log_stds_head = nn.Linear(
            hidden_dim, num_components * action_dim
        )

    def transition_features(self, z, z_next):
        scale = math.sqrt(2.0)
        midpoint = (z + z_next) / scale
        delta = (z_next - z) / scale
        interaction = self.u_m(midpoint) * self.u_d(delta)
        return midpoint, delta, interaction

    def distribution(self, z, z_next, prev_act_emb):
        midpoint, delta, interaction = self.transition_features(z, z_next)
        features = torch.cat(
            [midpoint, delta, interaction, prev_act_emb], dim=-1
        )
        hidden = self.trunk(features)
        logits = self.logits_head(hidden)
        log_weights = torch.log_softmax(logits, dim=-1)
        weights = log_weights.exp()
        out_shape = (*hidden.shape[:-1], self.num_components, self.action_dim)
        means = self.means_head(hidden).reshape(out_shape)
        log_stds = self.log_stds_head(hidden).reshape(out_shape).clamp(
            self.min_log_std, self.max_log_std
        )
        expectation_mean = (weights.unsqueeze(-1) * means).sum(dim=-2)
        map_component = weights.argmax(dim=-1)
        gather_index = map_component[..., None, None].expand(
            *map_component.shape, 1, self.action_dim
        )
        map_mean = means.gather(dim=-2, index=gather_index).squeeze(-2)
        map_log_std = log_stds.gather(dim=-2, index=gather_index).squeeze(-2)
        return {
            "kind": "mixture_gaussian",
            "logits": logits,
            "log_weights": log_weights,
            "weights": weights,
            "means": means,
            "log_stds": log_stds,
            "map_component": map_component,
            "map_mean": map_mean,
            "map_log_std": map_log_std,
            "expectation_mean": expectation_mean,
        }

    def forward(self, z, z_next, prev_act_emb):
        stats = self.distribution(z, z_next, prev_act_emb)
        return stats["map_mean"], stats["map_log_std"]

    def action_mean(self, z, z_next, prev_act_emb):
        stats = self.distribution(z, z_next, prev_act_emb)
        if self.planning_mode == "mog_expectation":
            return stats["expectation_mean"]
        return stats["map_mean"]


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
