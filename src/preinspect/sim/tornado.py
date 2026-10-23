"""Scenario runs and one-at-a-time sensitivity (tornado) of the line simulation."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from preinspect.sim.line import OUTCOMES, ROUTES, simulate


def route_matrix(route_confusion: dict[str, dict[str, int]]) -> dict[str, np.ndarray]:
    """P(route | true outcome) from a route-by-outcome count table."""
    out = {}
    for o in OUTCOMES:
        counts = np.array([route_confusion.get(r, {}).get(o, 0) for r in ROUTES], dtype=float)
        out[o] = counts / counts.sum() if counts.sum() else np.array([1.0, 0, 0, 0])
    return out


def outcome_mix(route_confusion: dict[str, dict[str, int]]) -> np.ndarray:
    """Share of each true outcome in the evaluated population."""
    totals = np.array(
        [sum(route_confusion.get(r, {}).get(o, 0) for r in ROUTES) for o in OUTCOMES], dtype=float
    )
    return totals / totals.sum()


def adjust(
    matrix: dict[str, np.ndarray],
    scrap_capture: float | None = None,
    false_quarantine: float | None = None,
) -> dict[str, np.ndarray]:
    """Change the scrap-capture or false-quarantine rate, keeping other proportions."""
    m = {k: v.copy() for k, v in matrix.items()}
    if scrap_capture is not None:
        s = m["SCRAP"]
        caught, missed = s[2] + s[3], s[0] + s[1]
        s[2:] = s[2:] / caught * scrap_capture if caught else [scrap_capture, 0]
        s[:2] = s[:2] / missed * (1 - scrap_capture) if missed else [1 - scrap_capture, 0]
    if false_quarantine is not None:
        for o in ("PASS", "REWORK"):
            row = m[o]
            others = np.delete(np.arange(4), 2)
            rest = row[others].sum()
            row[others] = row[others] / rest * (1 - false_quarantine) if rest else 0
            row[2] = false_quarantine
    return m


def compare(
    sim: dict[str, Any],
    costs: dict[str, Any],
    mix: np.ndarray,
    matrix: dict[str, np.ndarray],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """Baseline and gated line, ``reps`` replications each, common random numbers."""
    rows = []
    for rep in range(reps):
        for gated in (False, True):
            k = simulate(sim, costs, mix, matrix, gated, seed + rep)
            rows.append({"scenario": "with gate" if gated else "baseline", "replication": rep, **k})
    return pd.DataFrame(rows)


def saving(
    sim: dict[str, Any],
    costs: dict[str, Any],
    mix: np.ndarray,
    matrix: dict[str, np.ndarray],
    reps: int,
    seed: int,
) -> float:
    """Mean cost-per-unit saving of the gate over the baseline."""
    df = compare(sim, costs, mix, matrix, reps, seed)
    means = df.groupby("scenario")["cost_per_unit"].mean()
    return float(means["baseline"] - means["with gate"])


def tornado(
    sim: dict[str, Any],
    costs: dict[str, Any],
    mix: np.ndarray,
    matrix: dict[str, np.ndarray],
    seed: int,
) -> pd.DataFrame:
    """Saving per unit at the low and high value of each uncertain input."""
    t = sim["tornado"]
    reps = t["replications"]
    base = saving(sim, costs, mix, matrix, reps, seed)
    rows = []
    for name in (
        "scrap_capture",
        "false_quarantine_rate",
        "arrivals_per_day",
        "standard_clean_cost",
        "wrongful_scrap_value",
    ):
        result = {}
        for label, value in zip(("low", "high"), t[name], strict=True):
            s, c, m = copy.deepcopy(sim), copy.deepcopy(costs), matrix
            if name == "scrap_capture":
                m = adjust(matrix, scrap_capture=value)
            elif name == "false_quarantine_rate":
                m = adjust(matrix, false_quarantine=value)
            elif name == "arrivals_per_day":
                s["arrivals"]["units_per_day"] = value
            elif name == "standard_clean_cost":
                c["items"]["standard_clean"]["value"] = value
            else:
                c["items"]["wrongful_scrap_value"]["value"] = value
            result[label] = (value, saving(s, c, mix, m, reps, seed))
        rows.append(
            {
                "parameter": name,
                "low_value": result["low"][0],
                "high_value": result["high"][0],
                "saving_low": result["low"][1],
                "saving_high": result["high"][1],
                "saving_base": base,
            }
        )
    df = pd.DataFrame(rows)
    df["swing"] = (df["saving_high"] - df["saving_low"]).abs()
    return df.sort_values("swing", ascending=False)
