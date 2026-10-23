"""Incoming flow-bench curve: pressure drop against volumetric flow.

``dP = a*Q + b*Q**2``. The linear term represents flow through the porous
walls and deposit layers; the quadratic term represents inertial losses at
channel entries. Loading raises both terms; leak paths (cracks, missing
plugs, melt) let flow bypass the walls and lower the measured pressure drop.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def geometry_factors(gcfg: dict[str, Any], meta: dict[str, Any]) -> tuple[float, float]:
    """Scale factors for the clean-part ``a`` and ``b`` coefficients.

    The wall term scales inversely with filtration area (diameter squared times
    length); the inertial term scales with the inverse fourth power of diameter.
    """
    f, sub = gcfg["flow"], gcfg["substrate"]
    d_ratio = sub["reference_diameter_mm"] / meta["diameter_mm"]
    l_ratio = f["reference_length_mm"] / meta["length_mm"]
    cpsi_ratio = (meta["cell_density_cpsi"] / 200.0) ** f["cpsi_exponent"]
    return d_ratio**2 * l_ratio * cpsi_ratio, d_ratio**4


def new_part_dp_kpa(gcfg: dict[str, Any], meta: dict[str, Any]) -> float:
    """Pressure drop of a new part of this geometry at the rated flow (kPa)."""
    f = gcfg["flow"]
    ga, gb = geometry_factors(gcfg, meta)
    q = f["rated_q_m3s"]
    return f["a0_kpa_s_per_m3"] * ga * q + f["b0_kpa_s2_per_m6"] * gb * q**2


def flow_curve(
    gcfg: dict[str, Any],
    meta: dict[str, Any],
    cond: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate the incoming flow-bench measurement.

    Parameters
    ----------
    gcfg : dict
        Generator configuration.
    meta, cond : dict
        Unit metadata and latent condition.
    rng : numpy.random.Generator
        Unit generator.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2, n_points)``: row 0 flow in m3/s, row 1 pressure
        drop in kPa, with NaN where the sensor dropped out.
    """
    f = gcfg["flow"]
    q = np.asarray(f["q_m3s"], dtype=float)
    ga, gb = geometry_factors(gcfg, meta)
    coke_or_soot_face = cond["face_plug_extent"]
    load_a = (
        1.0
        + f["soot_a_per_gL"] * cond["soot_load_gL"]
        + f["ash_a_per_gL"] * cond["ash_load_gL"]
        + f["face_plug_a"] * coke_or_soot_face
        + f["glazing_a"] * cond["glazing"]
    )
    load_b = 1.0 + f["ash_b_per_gL"] * cond["ash_load_gL"] + f["face_plug_a"] * coke_or_soot_face
    a = f["a0_kpa_s_per_m3"] * ga * load_a
    b = f["b0_kpa_s2_per_m6"] * gb * load_b
    dp = (a * q + b * q**2) * (1.0 - cond["bypass_fraction"]) ** f["bypass_exponent"]
    dp = dp * (1.0 + rng.normal(0.0, f["noise_rel"], q.size)) + rng.normal(
        0.0, f["noise_abs_kpa"], q.size
    )
    dropout = rng.random(q.size) < f["point_dropout"]
    if rng.random() < f["sensor_fault_rate"]:
        lo, hi = f["sensor_fault_points"]
        k = int(rng.integers(lo, hi + 1))
        start = int(rng.integers(0, q.size - k + 1))
        dropout[start : start + k] = True
    dp[dropout] = np.nan
    return np.stack([q, dp]).astype(np.float32)
