"""Device selection (CUDA > MPS > CPU)."""
from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    On this project's dev machine (Apple Silicon) this resolves to MPS;
    heavy training on the cluster will resolve to CUDA.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
