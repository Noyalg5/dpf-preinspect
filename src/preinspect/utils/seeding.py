"""Deterministic seeding.

Every random draw in the project comes from a generator derived from the
global seed plus a string key, so adding a new consumer of randomness does
not shift the streams used by existing ones.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def derive_seed(seed: int, *keys: object) -> int:
    """Derive a stable 32-bit child seed from a parent seed and keys.

    Parameters
    ----------
    seed : int
        Parent seed.
    *keys : object
        Any values whose string form identifies the consumer, e.g. ``("unit", 17)``.

    Returns
    -------
    int
        Child seed in ``[0, 2**32)``.
    """
    text = ":".join([str(seed), *map(str, keys)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def make_rng(seed: int, *keys: object) -> np.random.Generator:
    """Return a NumPy generator seeded from ``derive_seed(seed, *keys)``."""
    return np.random.default_rng(derive_seed(seed, *keys))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch global generators.

    Parameters
    ----------
    seed : int
        Global seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Third-party code may still draw from the legacy global generator.
    np.random.seed(seed % (2**32))  # noqa: NPY002
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
