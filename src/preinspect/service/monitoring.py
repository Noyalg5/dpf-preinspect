"""Monitoring calculations for the quality-engineer dashboard.

- Population stability index (PSI) of recent inputs against the training
  reference, per metadata field and per image-embedding component.
  Conventional reading: below 0.10 stable, 0.10 to 0.25 worth watching,
  above 0.25 a material shift.
- Shewhart p-chart of the REFER + QUARANTINE share per batch of inspections.
- Calibration over time: mean predicted probability of the recorded outcome
  against the observed hit rate, once downstream results are logged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index using quantile bins of the reference."""
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]
    if ref.size < bins or cur.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / ref.size
    c = np.histogram(cur, edges)[0] / cur.size
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_table(
    reference: pd.DataFrame, current: pd.DataFrame, bins: int, warn: float, alert: float
) -> list[dict[str, Any]]:
    """PSI and status for every column present in both tables."""
    reference = reference.apply(pd.to_numeric, errors="coerce")
    current = current.apply(pd.to_numeric, errors="coerce")
    rows = []
    for col in reference.columns:
        if col not in current:
            continue
        value = psi(reference[col].to_numpy(float), current[col].to_numpy(float), bins)
        status = (
            "n/a"
            if not np.isfinite(value)
            else "alert"
            if value > alert
            else "watch"
            if value > warn
            else "stable"
        )
        rows.append(
            {
                "feature": col,
                "psi": value,
                "status": status,
                "reference_mean": float(np.nanmean(reference[col])),
                "current_mean": float(np.nanmean(current[col])),
            }
        )
    return rows


def p_chart(routes: pd.Series, batch: int) -> pd.DataFrame:
    """Share of REFER or QUARANTINE per consecutive batch, with 3-sigma limits."""
    flagged = routes.isin(["REFER", "QUARANTINE"]).to_numpy(float)
    n_batches = len(flagged) // batch
    if n_batches == 0:
        return pd.DataFrame(columns=["batch", "p", "centre", "lcl", "ucl", "out_of_control"])
    p = flagged[: n_batches * batch].reshape(n_batches, batch).mean(axis=1)
    centre = float(p.mean())
    sigma = np.sqrt(centre * (1 - centre) / batch)
    lcl, ucl = max(0.0, centre - 3 * sigma), min(1.0, centre + 3 * sigma)
    return pd.DataFrame(
        {
            "batch": np.arange(1, n_batches + 1),
            "p": p,
            "centre": centre,
            "lcl": lcl,
            "ucl": ucl,
            "out_of_control": (p > ucl) | (p < lcl),
        }
    )


def calibration_over_time(
    inspections: pd.DataFrame, results: pd.DataFrame, batch: int
) -> pd.DataFrame:
    """Per batch: mean predicted probability of the recorded outcome versus hit rate.

    Parameters
    ----------
    inspections : pandas.DataFrame
        ``unit_id`` and ``probabilities`` (dict) per inspection, in time order.
    results : pandas.DataFrame
        ``unit_id`` and ``disposition`` from downstream tests.
    """
    df = inspections.merge(results, on="unit_id", how="inner")
    if df.empty:
        return pd.DataFrame(columns=["batch", "confidence", "accuracy", "brier"])
    top = df["probabilities"].map(lambda p: max(p, key=p.get))
    conf = df["probabilities"].map(lambda p: max(p.values()))
    hit = (top == df["disposition"]).astype(float)
    brier = df.apply(
        lambda r: sum(
            (r["probabilities"][k] - (k == r["disposition"])) ** 2 for k in r["probabilities"]
        ),
        axis=1,
    )
    groups = np.arange(len(df)) // batch
    out = pd.DataFrame(
        {
            "batch": groups + 1,
            "confidence": conf.to_numpy(),
            "accuracy": hit.to_numpy(),
            "brier": brier.to_numpy(),
        }
    )
    return out.groupby("batch", as_index=False).mean()
