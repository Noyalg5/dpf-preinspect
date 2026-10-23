"""Read access to a generated dataset (images, masks, signals, tables)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def read_image(ddir: Path, unit_id: str, view: str) -> np.ndarray:
    """Read a face image as RGB uint8."""
    bgr = cv2.imread(str(ddir / "images" / f"{unit_id}_{view}.jpg"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"image missing for {unit_id} {view}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(ddir: Path, unit_id: str, view: str) -> np.ndarray:
    """Read a defect bitmask (uint8, bit k = class k)."""
    mask = cv2.imread(str(ddir / "masks" / f"{unit_id}_{view}.png"), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"mask missing for {unit_id} {view}")
    return mask


def decode_bitmask(mask: np.ndarray, n_classes: int) -> np.ndarray:
    """Expand a bitmask to a boolean array of shape ``(n_classes, H, W)``."""
    bits = np.arange(n_classes, dtype=np.uint8)
    return ((mask[None, :, :] >> bits[:, None, None]) & 1).astype(bool)


def encode_bitmask(masks: np.ndarray) -> np.ndarray:
    """Pack a boolean ``(n_classes, H, W)`` array into a uint8 bitmask."""
    out = np.zeros(masks.shape[1:], dtype=np.uint8)
    for k in range(masks.shape[0]):
        out |= (masks[k].astype(np.uint8) << k).astype(np.uint8)
    return out


@lru_cache(maxsize=4)
def load_signals(ddir: Path) -> np.ndarray:
    """Flow curves, shape ``(N, 2, n_points)`` (row 0 flow m3/s, row 1 dP kPa)."""
    return np.load(ddir / "signals.npy")


def flow_curve(ddir: Path, signal_index: int) -> np.ndarray:
    """One unit's flow curve, shape ``(2, n_points)``."""
    return load_signals(ddir)[int(signal_index)]


def load_latent(ddir: Path) -> pd.DataFrame:
    """Latent truth table (evaluation only, never a model input)."""
    return pd.read_parquet(ddir / "latent.parquet")


def load_scenes(ddir: Path) -> dict[str, dict[str, Any]]:
    """Scene, acquisition and render records keyed by unit ID."""
    scenes = {}
    with open(ddir / "scenes.jsonl", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            scenes[rec["unit_id"]] = rec
    return scenes
