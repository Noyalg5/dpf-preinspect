"""Data-quality fault injection.

Faults imitate what goes wrong in real goods-in records: blank fields,
values entered in the wrong unit or scale, the same unit booked in twice,
temperatures logged in Fahrenheit, and dispositions mistyped. Every injected
fault is written to ``faults.parquet`` so the validation gate can be tested
against ground truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from preinspect.utils.seeding import make_rng

MISSING_COLUMNS = ["operating_hours", "weight_gain_g", "application", "material"]
OUTCOMES = ["PASS", "REWORK", "SCRAP"]


def _out_of_range(row: pd.Series, rng: np.random.Generator) -> tuple[str, Any]:
    """Pick a column and a plausible data-entry error that breaks its range."""
    choice = int(rng.integers(0, 4))
    if choice == 0:
        return "operating_hours", -abs(row["operating_hours"])
    if choice == 1:
        return "weight_gain_g", row["weight_gain_g"] * 10.0 + 3000.0
    if choice == 2:
        return "diameter_mm", row["diameter_mm"] / 10.0
    return "previous_cleans", -1


def inject_faults(
    units: pd.DataFrame, gcfg: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject configured faults into a clean units table.

    Parameters
    ----------
    units : pandas.DataFrame
        Clean units table (one row per unit).
    gcfg : dict
        Generator configuration (``faults`` section gives the rates).
    seed : int
        Global seed.

    Returns
    -------
    units : pandas.DataFrame
        Table with faults applied (duplicates appended as extra rows).
    faults : pandas.DataFrame
        Ground truth: ``unit_id``, ``fault_type``, ``column``, ``original``, ``injected``.
    """
    rng = make_rng(seed, "faults")
    rates = gcfg["faults"]
    out = units.copy()
    out["max_regen_temp_c"] = out["max_regen_temp_c"].astype(float)
    out["operating_hours"] = out["operating_hours"].astype(float)
    out["previous_cleans"] = out["previous_cleans"].astype(float)
    out["application"] = out["application"].astype(object)
    out["material"] = out["material"].astype(object)
    n = len(out)
    order = rng.permutation(n)
    counts = {k: int(round(rate * n)) for k, rate in rates.items()}
    log: list[dict[str, Any]] = []
    cursor = 0

    def take(k: int) -> np.ndarray:
        nonlocal cursor
        idx = order[cursor : cursor + k]
        cursor += k
        return idx

    for i in take(counts["missing_value"]):
        col = MISSING_COLUMNS[int(rng.integers(0, len(MISSING_COLUMNS)))]
        log.append(
            {
                "row": i,
                "fault_type": "missing_value",
                "column": col,
                "original": out.at[i, col],
                "injected": None,
            }
        )
        out.at[i, col] = None if out[col].dtype == object else np.nan
    for i in take(counts["out_of_range"]):
        col, value = _out_of_range(out.loc[i], rng)
        log.append(
            {
                "row": i,
                "fault_type": "out_of_range",
                "column": col,
                "original": out.at[i, col],
                "injected": value,
            }
        )
        out.at[i, col] = value
    logged = np.flatnonzero(out["max_regen_temp_c"].notna().to_numpy())
    candidates = np.setdiff1d(logged, order[:cursor])
    mixups = rng.choice(
        candidates, size=min(counts["unit_mixup_f"], len(candidates)), replace=False
    )
    used = set(order[:cursor].tolist()) | set(mixups.tolist())
    for i in mixups:
        celsius = out.at[i, "max_regen_temp_c"]
        fahrenheit = round(celsius * 9.0 / 5.0 + 32.0, 1)
        log.append(
            {
                "row": i,
                "fault_type": "unit_mixup_f",
                "column": "max_regen_temp_c",
                "original": celsius,
                "injected": fahrenheit,
            }
        )
        out.at[i, "max_regen_temp_c"] = fahrenheit
    remaining = [i for i in order if i not in used]
    flips = remaining[: counts["label_flip"]]
    for i in flips:
        original = out.at[i, "disposition"]
        new = str(rng.choice([o for o in OUTCOMES if o != original]))
        log.append(
            {
                "row": i,
                "fault_type": "label_flip",
                "column": "disposition",
                "original": original,
                "injected": new,
            }
        )
        out.at[i, "disposition"] = new
    dup_sources = remaining[counts["label_flip"] : counts["label_flip"] + counts["duplicate_id"]]
    duplicates = []
    for i in dup_sources:
        dup = out.loc[i].copy()
        dup["arrival_date"] = (
            (pd.Timestamp(dup["arrival_date"]) + pd.Timedelta(days=int(rng.integers(1, 30))))
            .date()
            .isoformat()
        )
        dup["weight_gain_g"] = round(float(dup["weight_gain_g"]) + rng.normal(0, 40), 1)
        duplicates.append(dup)
        log.append(
            {
                "row": i,
                "fault_type": "duplicate_id",
                "column": "unit_id",
                "original": dup["unit_id"],
                "injected": dup["unit_id"],
            }
        )
    if duplicates:
        out = pd.concat([out, pd.DataFrame(duplicates)], ignore_index=True)
    faults = pd.DataFrame(log)
    faults.insert(0, "unit_id", [units.at[r, "unit_id"] for r in faults["row"]])
    for col in ("original", "injected"):
        faults[col] = faults[col].astype(str)
    return out, faults.drop(columns="row")
