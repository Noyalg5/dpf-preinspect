"""Downstream outcome model: latent condition to test results and disposition.

Every unit is assumed to receive the standard thermal and pneumatic clean,
then a flow-restriction test (pressure-drop ratio against the new-part
specification) and a filtration-integrity test. Units that fail only on
restriction receive an extended clean and are re-tested.

The function returns both the *true* values (latent, used for evaluation)
and the *measured* values that a real line would record (test noise and
occasional integrity-test errors included).
"""

from __future__ import annotations

from typing import Any

import numpy as np

OUTCOMES = ("PASS", "REWORK", "SCRAP")


def _disposition(integrity: bool, dp_std: float, dp_ext: float, spec: float) -> str:
    if not integrity:
        return "SCRAP"
    if dp_std <= spec:
        return "PASS"
    return "REWORK" if dp_ext <= spec else "SCRAP"


def integrity_ok(ocfg: dict[str, Any], cond: dict[str, Any]) -> bool:
    """True if the unit has no leak path severe enough to fail the integrity test."""
    crack = max((c["severity"] for c in cond["cracks"]), default=0.0)
    plugs = cond["missing_plugs_inlet"] + cond["missing_plugs_outlet"]
    return bool(
        crack <= ocfg["crack_fail_severity"]
        and cond["internal_crack_severity"] <= ocfg["crack_fail_severity"]
        and plugs <= ocfg["missing_plug_fail"]
        and cond["chip_severity"] <= ocfg["chip_fail_severity"]
        and cond["melt_severity"] == 0.0
    )


def downstream_outcome(
    gcfg: dict[str, Any], cond: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    """Simulate cleaning and the downstream tests for one unit.

    Parameters
    ----------
    gcfg : dict
        Generator configuration.
    cond : dict
        Latent condition.
    rng : numpy.random.Generator
        Unit generator.

    Returns
    -------
    dict
        ``*_true`` fields (latent) and measured fields: ``dp_ratio_post_clean``,
        ``dp_ratio_post_extended`` (NaN unless an extended clean was done),
        ``integrity_pass`` and ``disposition``.
    """
    o = gcfg["outcome"]
    glazing = cond["glazing"]
    eff_std = float(
        np.clip(
            o["std_clean_eff"] - o["std_eff_glazing"] * glazing + rng.normal(0, o["std_eff_sd"]),
            0.1,
            0.98,
        )
    )
    residual_std = cond["ash_load_gL"] * (1.0 - eff_std)
    residual_ext = residual_std * (
        1.0 - o["ext_recovery"] * (1.0 - o["ext_glazing_block"] * glazing)
    )
    coke = cond["face_plug_extent"] if cond["face_plug_deposit"] == "coke" else 0.0
    base = (
        1.0
        + o["dp_glazing"] * glazing
        + o["dp_coke"] * coke
        + o["dp_melt"] * cond["melt_severity"]
        - o["dp_bypass"] * cond["bypass_fraction"]
    )
    dp_std_true = base + o["dp_per_residual_ash_gL"] * residual_std
    dp_ext_true = base + o["dp_per_residual_ash_gL"] * residual_ext
    integrity_true = integrity_ok(o, cond)
    spec = o["spec_ratio"]

    # The disposition is decided on the values as recorded (4 decimals), so the
    # record is always self-consistent.
    dp_std = round(float(dp_std_true + rng.normal(0.0, o["dp_noise_sd"])), 4)
    dp_ext = round(float(dp_ext_true + rng.normal(0.0, o["dp_noise_sd"])), 4)
    flip = o["integrity_false_fail"] if integrity_true else o["integrity_false_pass"]
    integrity = integrity_true != bool(rng.random() < flip)
    disposition = _disposition(integrity, dp_std, dp_ext, spec)
    extended_done = integrity and dp_std > spec
    return {
        "dp_ratio_post_clean_true": float(dp_std_true),
        "dp_ratio_post_extended_true": float(dp_ext_true),
        "integrity_true": integrity_true,
        "outcome_true": _disposition(integrity_true, dp_std_true, dp_ext_true, spec),
        "residual_ash_std_gL": float(residual_std),
        "dp_ratio_post_clean": dp_std,
        "dp_ratio_post_extended": dp_ext if extended_done else float("nan"),
        "integrity_pass": bool(integrity),
        "disposition": disposition,
    }


def disposition_rule(
    integrity: bool, dp_post_clean: float, dp_post_extended: float, spec: float
) -> str:
    """Disposition implied by recorded test results (used by the validation gate)."""
    ext = dp_post_extended if np.isfinite(dp_post_extended) else np.inf
    return _disposition(integrity, dp_post_clean, ext, spec)
