"""Explicit scaled-dot-product-attention backend policy."""

from __future__ import annotations

from contextlib import nullcontext

import torch


VALID_SDPA_BACKENDS = ("auto", "math")


def validate_sdpa_backend(backend: str) -> str:
    """Normalize and validate a user-facing SDPA policy name."""
    normalized = str(backend).strip().lower()
    if normalized not in VALID_SDPA_BACKENDS:
        expected = ", ".join(VALID_SDPA_BACKENDS)
        raise ValueError(
            f"sdpa_backend must be one of [{expected}], got {backend!r}"
        )
    return normalized


def sdpa_kernel_context(backend: str):
    """Return a context that applies the requested SDPA kernel policy.

    ``auto`` leaves PyTorch's dispatcher untouched. ``math`` permits only the
    math implementation, including attention calls made by the ViT encoder.
    """
    backend = validate_sdpa_backend(backend)
    if backend == "auto":
        return nullcontext()
    return torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)


def sdpa_policy_metadata(backend: str) -> dict[str, object]:
    """Build an auditable description of the requested runtime policy."""
    backend = validate_sdpa_backend(backend)
    return {
        "requested": backend,
        "effective_policy": (
            "pytorch_auto_dispatch" if backend == "auto" else "math_only"
        ),
        "torch_version": torch.__version__,
    }
