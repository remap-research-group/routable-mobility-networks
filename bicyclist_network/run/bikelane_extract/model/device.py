"""Pick a torch device: CUDA → Apple MPS → CPU. Mixed precision only on CUDA."""
from __future__ import annotations

import contextlib


def pick(explicit: str | None = None) -> str:
    import torch
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast(device: str):
    """torch.autocast on CUDA; a no-op elsewhere (MPS/CPU run in float32)."""
    import torch
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda")
    return contextlib.nullcontext()


def describe(device: str) -> str:
    import torch
    if device.startswith("cuda"):
        return f"{device} ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return "mps (Apple GPU)"
    return "cpu"
