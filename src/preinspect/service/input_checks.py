"""Input validation at inspection time.

The same rules as the data-validation gate, applied to one unit. Any
failure routes the unit to REFER with the reasons listed, rather than
letting the models guess on bad inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

VIEWS = ("inlet", "outlet")


def check_unit(
    meta: dict[str, Any],
    images: dict[str, np.ndarray],
    curve: np.ndarray,
    lighting: dict[str, str],
    cfg: dict[str, Any],
) -> list[str]:
    """Return a list of human-readable problems (empty if the inputs are usable)."""
    v = cfg["validation"]
    issues = []
    for col in v["required"]:
        if col in ("unit_id", "arrival_date"):
            continue
        if meta.get(col) is None or (isinstance(meta.get(col), float) and np.isnan(meta[col])):
            issues.append(f"{col} is missing")
    for col, (lo, hi) in v["ranges"].items():
        val = meta.get(col)
        if isinstance(val, int | float) and np.isfinite(val) and not lo <= val <= hi:
            issues.append(f"{col}={val} is outside the plausible range {lo} to {hi}")
    temp = meta.get("max_regen_temp_c")
    if isinstance(temp, int | float) and np.isfinite(temp) and temp > v["fahrenheit_suspect_above"]:
        issues.append(f"max_regen_temp_c={temp} may be in Fahrenheit; please verify")
    allowed = {
        "material": cfg["generator"]["substrate"]["materials"],
        "application": cfg["generator"]["applications"],
    }
    for col, values in allowed.items():
        if meta.get(col) is not None and meta[col] not in values:
            issues.append(f"{col}={meta[col]!r} is not a known value")
    size = cfg["generator"]["image_size"]
    for view in VIEWS:
        img = images.get(view)
        if img is None or img.ndim != 3 or img.shape[2] != 3:
            issues.append(f"{view} image is missing or not RGB")
        elif img.shape[:2] != (size, size):
            issues.append(
                f"{view} image is {img.shape[1]}x{img.shape[0]} px; expected {size}x{size}"
            )
        elif float(img.std()) < v["image_min_std"]:
            issues.append(f"{view} image is blank or saturated")
        if lighting.get(view) not in cfg["generator"]["acquisition"]["lighting"]:
            issues.append(f"{view} lighting mode {lighting.get(view)!r} is not recognised")
    n_points = len(cfg["generator"]["flow"]["q_m3s"])
    if curve is None or np.shape(curve) != (2, n_points):
        issues.append(f"flow curve must have shape (2, {n_points})")
    elif np.isnan(curve[1]).sum() > v["max_flow_dropouts"]:
        issues.append("flow curve has too many missing points")
    return issues
