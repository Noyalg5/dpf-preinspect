"""Per-cell state classification: normal, missing plug or blocked.

Each cell centre is sampled after the lattice is known. Under front light a
plugged cell is light and an open channel dark, so:

- a cell that *should* be plugged but reads as dark as its open neighbours
  has lost its plug;
- an open cell that reads light (ash fill) or sits where the plug/open
  contrast has collapsed (soot or coke crust) is blocked.

Under backlight only through-channels transmit light, so a missing plug is a
cell far brighter than the face median; blockage cannot be judged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import median_filter

from preinspect.classical.lattice import Lattice
from preinspect.classical.localise import Localisation

NORMAL, MISSING, BLOCKED = 0, 1, 2


@dataclass
class CellMap:
    """Sampled cells of one face."""

    i: np.ndarray
    j: np.ndarray
    x: np.ndarray
    y: np.ndarray
    plugged: np.ndarray
    value: np.ndarray
    state: np.ndarray
    contrast: float


def sample_cells(
    gray: np.ndarray, loc: Localisation, lat: Lattice, ccfg: dict[str, Any]
) -> CellMap:
    """Sample the mean grey level at every cell centre inside the face."""
    n = int(np.ceil(loc.radius / lat.pitch)) + 2
    i, j = np.meshgrid(np.arange(-n, n), np.arange(-n, n), indexing="ij")
    i, j = i.ravel(), j.ravel()
    x, y = lat.cell_centre(i, j, loc)
    inside = np.hypot(x - loc.cx, y - loc.cy) < ccfg["cells"]["face_r"] * loc.radius
    h, w = gray.shape
    inside &= (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    i, j, x, y = i[inside], j[inside], x[inside], y[inside]
    k = max(1, int(round(0.4 * lat.pitch)))
    smooth = cv2.blur(gray.astype(np.float32), (k, k))
    value = smooth[np.round(y).astype(int), np.round(x).astype(int)]
    plugged = np.mod(i + j, 2) == lat.plugged_parity
    return CellMap(i, j, x, y, plugged, value, np.zeros(i.size, np.int8), 0.0)


def _local_median(cm: CellMap, mask: np.ndarray, window: int) -> np.ndarray:
    """Median of the masked cells in a window around every cell.

    Cells outside the mask (the other checkerboard parity, or outside the
    face) are filled with the masked median first so a fast fixed-size
    median filter can be used.
    """
    i0, j0 = cm.i - cm.i.min(), cm.j - cm.j.min()
    fill = float(np.median(cm.value[mask])) if mask.any() else 0.0
    grid = np.full((i0.max() + 1, j0.max() + 1), fill, dtype=np.float32)
    grid[i0[mask], j0[mask]] = cm.value[mask]
    return median_filter(grid, size=window, mode="nearest")[i0, j0]


def _diagonal_count(cm: CellMap, flag: np.ndarray) -> np.ndarray:
    """Number of flagged cells among the four diagonal neighbours of each cell.

    In a checkerboard the nearest cells of the same parity are diagonal, so
    this measures whether a flagged open cell is part of a region (deposit)
    or an isolated line (a cement joint or the skin edge).
    """
    i0, j0 = cm.i - cm.i.min() + 1, cm.j - cm.j.min() + 1
    grid = np.zeros((i0.max() + 2, j0.max() + 2), dtype=np.int8)
    grid[i0[flag], j0[flag]] = 1
    return sum(grid[i0 + di, j0 + dj] for di in (-1, 1) for dj in (-1, 1))


def classify_cells(
    cm: CellMap, lighting: str, ccfg: dict[str, Any], exclude: np.ndarray | None = None
) -> CellMap:
    """Assign NORMAL, MISSING or BLOCKED to each cell in place and return it.

    ``exclude`` is a boolean image; cells whose centre falls on it (for
    example a detected crack) are never called missing plugs.
    """
    cfg = ccfg["cells"]
    if cm.value.size == 0:
        return cm
    on_excluded = (
        exclude[np.round(cm.y).astype(int), np.round(cm.x).astype(int)]
        if exclude is not None
        else np.zeros(cm.value.size, dtype=bool)
    )
    if lighting == "backlight":
        med = float(np.median(cm.value))
        mad = float(np.median(np.abs(cm.value - med))) + 1e-3
        cm.state[(cm.value > med + cfg["backlight_k"] * mad) & ~on_excluded] = MISSING
        cm.contrast = float("nan")
        return cm
    med_p = float(np.median(cm.value[cm.plugged]))
    med_o = float(np.median(cm.value[~cm.plugged]))
    cm.contrast = (med_p - med_o) / 255.0
    if cm.contrast < cfg["min_contrast"]:
        return cm
    loc_p = _local_median(cm, cm.plugged, cfg["window_cells"])
    loc_o = _local_median(cm, ~cm.plugged, cfg["window_cells"])
    c_loc = loc_p - loc_o
    c_glob = med_p - med_o
    low = c_loc < cfg["low_contrast"] * c_glob
    # A dark plug only counts as missing where its neighbouring plugs look
    # normal; a darkened neighbourhood is soot, not plug loss.
    plugs_normal = loc_p >= med_p - cfg["region_drop"] * c_glob
    missing = (
        cm.plugged
        & ~low
        & plugs_normal
        & ~on_excluded
        & (cm.value < loc_o + cfg["missing_level"] * c_loc)
    )
    filled = ~cm.plugged & (cm.value > med_o + cfg["blocked_level"] * c_glob)
    blocked = ~cm.plugged & (low | filled)
    blocked &= _diagonal_count(cm, blocked) >= cfg["blocked_min_neighbours"]
    cm.state[missing] = MISSING
    cm.state[blocked] = BLOCKED
    return cm


def cell_summary(cm: CellMap) -> dict[str, float]:
    """Concepts from the cell map: counts and fractions."""
    n_open = int((~cm.plugged).sum())
    return {
        "cells_sampled": float(cm.value.size),
        "missing_plug_count": float((cm.state == MISSING).sum()),
        "blocked_cell_fraction": float((cm.state == BLOCKED).sum() / max(1, n_open)),
        "cell_contrast": cm.contrast,
    }
