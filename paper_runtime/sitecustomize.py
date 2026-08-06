"""Formal recovery process defaults loaded automatically by Python's site module."""

import os


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
