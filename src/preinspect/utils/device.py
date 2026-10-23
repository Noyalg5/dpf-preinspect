"""Compute-device selection (cuda, then mps, then cpu)."""

from __future__ import annotations

import os

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick the compute device.

    Parameters
    ----------
    preference : str
        ``"auto"`` tries cuda, then mps, then cpu. Any other value is used as
        given. The environment variable ``PREINSPECT_DEVICE`` overrides both,
        which lets CI force the CPU path.

    Returns
    -------
    torch.device
        Selected device.
    """
    preference = os.environ.get("PREINSPECT_DEVICE", preference)
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
