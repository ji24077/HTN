"""Automatic GPU selection with CPU fallback, respecting explicit device requests."""

import torch


def select_device(requested):
    if requested == "auto":
        if torch.cuda.is_available() and not torch.version.hip:
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and (not torch.cuda.is_available() or torch.version.hip):
        raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is disabled")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; CPU fallback is disabled")
    return torch.device(requested)
