"""Service-metadata features with explicit missing-value indicators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

NUMERIC = [
    "cell_density_cpsi",
    "diameter_mm",
    "length_mm",
    "operating_hours",
    "active_regen_count",
    "max_regen_temp_c",
    "weight_gain_g",
    "previous_cleans",
]
CATEGORICAL = ["material", "application"]


def metadata_features(units: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Numeric, derived and categorical metadata features.

    Derived features normalise by size and age: weight gain per litre of
    substrate and regenerations per 100 operating hours. Categorical columns
    use fixed category lists from config so encodings never depend on which
    values happen to appear in a split.
    """
    out = pd.DataFrame(index=units.index)
    for col in NUMERIC:
        out[f"meta_{col}"] = pd.to_numeric(units[col], errors="coerce").astype(float)
    volume_l = math.pi * (units["diameter_mm"] / 2.0) ** 2 * units["length_mm"] * 1e-6
    out["meta_volume_l"] = volume_l
    out["meta_load_g_per_l"] = units["weight_gain_g"] / volume_l
    out["meta_regens_per_100h"] = units["active_regen_count"] / (units["operating_hours"] / 100.0)
    out["meta_regen_temp_missing"] = units["max_regen_temp_c"].isna().astype(float)
    categories = {
        "material": list(cfg["generator"]["substrate"]["materials"]),
        "application": list(cfg["generator"]["applications"]),
    }
    for col in CATEGORICAL:
        out[f"meta_{col}"] = pd.Categorical(units[col], categories=categories[col])
    return out.replace([np.inf, -np.inf], np.nan)
