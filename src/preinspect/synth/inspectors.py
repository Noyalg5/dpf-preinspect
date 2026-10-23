"""Simulated human inspectors for the measurement system analysis (MSA).

Each inspector applies the same visual disposition rule as a trained
operator, but with a documented bias:

- **A** has a high detection threshold for cracks, so misses hairline cracks.
- **B** rejects stained units more often than the evidence warrants.
- **C** is sound early in a shift but drifts (higher crack threshold, more
  random errors) as the shift progresses.

Every inspector judges each MSA unit twice, at different points in a shift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from preinspect.utils.seeding import make_rng

OUTCOMES = ["PASS", "REWORK", "SCRAP"]


def visual_call(
    evidence: dict[str, Any],
    spec: dict[str, Any],
    rules: dict[str, Any],
    shift_position: float,
    rng: np.random.Generator,
) -> str:
    """One inspector's call on one unit.

    Parameters
    ----------
    evidence : dict
        What can be seen or read at the station: crack severity, chip
        severity, missing-plug count, face-plugging extent, stain type,
        outlet-soot extent, melt severity, weight gain per litre.
    spec : dict
        Inspector parameters (thresholds and bias).
    rules : dict
        Shared disposition rule thresholds.
    shift_position : float
        0 at the start of a shift, 1 at the end.
    rng : numpy.random.Generator
        Generator for perception noise and slips.

    Returns
    -------
    str
        ``PASS``, ``REWORK`` or ``SCRAP``.
    """
    drift = spec["drift"] * shift_position
    crack_seen = evidence["crack_severity"] * rng.lognormal(0, rules["perception_sd"])
    crack_min = spec["crack_min_severity"] * (1.0 + rules["drift_threshold_gain"] * drift)
    if (
        evidence["melt_severity"] > 0
        or (evidence["crack_severity"] > 0 and crack_seen > crack_min)
        or evidence["missing_plugs"] > rules["missing_plug_scrap"]
        or evidence["chip_severity"] > rules["chip_scrap"]
        or evidence["outlet_soot_extent"] > rules["outlet_soot_scrap"]
        or (evidence["stain"] != "none" and rng.random() < spec["stain_reject"])
    ):
        call = "SCRAP"
    elif (
        evidence["face_plug_extent"] > rules["face_plug_rework"]
        or evidence["load_g_per_l"] > rules["load_rework_g_per_l"]
        or evidence["stain"] == "coolant"
    ):
        call = "REWORK"
    else:
        call = "PASS"
    if rng.random() < spec["flip"] * (1.0 + rules["drift_slip_gain"] * drift):
        call = str(rng.choice([o for o in OUTCOMES if o != call]))
    return call


def simulate_inspectors(
    msa_ids: list[str],
    units: pd.DataFrame,
    latent: pd.DataFrame,
    gcfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Two trials per inspector over the MSA units.

    Returns
    -------
    pandas.DataFrame
        Columns ``unit_id``, ``inspector``, ``trial``, ``shift_position``, ``call``.
    """
    u = units.set_index("unit_id")
    lat = latent.set_index("unit_id")
    rows = []
    for name, spec in gcfg["inspectors"].items():
        for trial in (1, 2):
            rng = make_rng(seed, "inspector", name, trial)
            positions = rng.permutation(len(msa_ids)) / max(1, len(msa_ids) - 1)
            for uid, pos in zip(msa_ids, positions, strict=True):
                lt = lat.loc[uid]
                evidence = {
                    "crack_severity": lt["crack_max_severity"],
                    "melt_severity": lt["melt_severity"],
                    "missing_plugs": lt["missing_plugs_inlet"] + lt["missing_plugs_outlet"],
                    "chip_severity": lt["chip_severity"],
                    "outlet_soot_extent": lt["outlet_soot_extent"],
                    "face_plug_extent": lt["face_plug_extent"],
                    "stain": lt["stain"],
                    "load_g_per_l": u.loc[uid, "weight_gain_g"] / lt["volume_l"],
                }
                call = visual_call(evidence, spec, gcfg["inspector_rules"], float(pos), rng)
                rows.append(
                    {
                        "unit_id": uid,
                        "inspector": name,
                        "trial": trial,
                        "shift_position": round(float(pos), 3),
                        "call": call,
                    }
                )
    return pd.DataFrame(rows)
