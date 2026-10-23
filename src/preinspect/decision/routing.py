"""Configurable routing policy and cost-based threshold selection.

Rules, in order (the first that applies wins):

1. input validation failed            -> REFER
2. unexplained anomaly above threshold -> REFER
3. P(SCRAP) >= t_quarantine           -> QUARANTINE (a person confirms; never automatic scrap)
4. SCRAP in the conformal set          -> REFER (scrap cannot be ruled out)
5. P(REWORK) >= t_rework              -> EXTENDED_CLEAN
6. otherwise                          -> PROCEED
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from preinspect.decision.cost import expected_cost, route_rates

REASONS = {
    "invalid": "input validation failed",
    "anomaly": "unrecognised anomaly on a face",
    "quarantine": "scrap likely",
    "ambiguous": "scrap cannot be ruled out",
    "rework": "restriction likely above specification after a standard clean",
    "proceed": "no indication against a standard clean",
}


@dataclass
class Policy:
    """Routing thresholds."""

    t_quarantine: float
    t_rework: float
    alpha: float
    anomaly_threshold: float

    def as_dict(self) -> dict[str, float]:
        """Plain dictionary for JSON."""
        return asdict(self)


def route(
    proba: np.ndarray,
    scrap_in_set: np.ndarray,
    anomaly: np.ndarray,
    valid: np.ndarray,
    policy: Policy,
) -> tuple[np.ndarray, np.ndarray]:
    """Routes and reason codes for a batch of units.

    Parameters
    ----------
    proba : numpy.ndarray
        ``(n, 3)`` calibrated probabilities (PASS, REWORK, SCRAP).
    scrap_in_set : numpy.ndarray of bool
        Whether SCRAP is in each unit's conformal set.
    anomaly : numpy.ndarray
        Unexplained-anomaly score per unit (NaN treated as zero).
    valid : numpy.ndarray of bool
        Input validation result.
    policy : Policy
        Thresholds.
    """
    anomalous = np.nan_to_num(anomaly, nan=0.0) > policy.anomaly_threshold
    conditions = [
        ~valid,
        anomalous,
        proba[:, 2] >= policy.t_quarantine,
        scrap_in_set,
        proba[:, 1] >= policy.t_rework,
    ]
    routes = np.select(
        conditions, ["REFER", "REFER", "QUARANTINE", "REFER", "EXTENDED_CLEAN"], default="PROCEED"
    )
    reasons = np.select(
        conditions, ["invalid", "anomaly", "quarantine", "ambiguous", "rework"], default="proceed"
    )
    return routes, reasons


def _grid(spec: list[float]) -> np.ndarray:
    lo, hi, step = spec
    return np.round(np.arange(lo, hi + step / 2, step), 4)


def choose_thresholds(
    proba: np.ndarray,
    scrap_in_set: dict[float, np.ndarray],
    anomaly: np.ndarray,
    outcomes: np.ndarray,
    matrix: np.ndarray,
    cap: float,
    rcfg: dict,
    anomaly_threshold: float,
    min_capture: float = 0.0,
) -> tuple[Policy, pd.DataFrame]:
    """Grid-search thresholds and the routing alpha minimising expected cost.

    The wrongful-scrap cap and the minimum scrap capture are constraints;
    if both cannot be met, capture is maximised within the cap. ``scrap_in_set`` gives, for
    each candidate alpha, whether SCRAP is in each unit's conformal set; a
    smaller alpha means larger sets and more referrals, which the cost model
    weighs against missed scrap.

    Returns
    -------
    policy : Policy
        Cheapest feasible setting (the most cautious one if none is feasible).
    search : pandas.DataFrame
        Cost and safety rates for every grid point.
    """
    valid = np.ones(len(proba), dtype=bool)
    rows = []
    for alpha, in_set in scrap_in_set.items():
        for tq in _grid(rcfg["t_quarantine_grid"]):
            for tr in _grid(rcfg["t_rework_grid"]):
                pol = Policy(float(tq), float(tr), float(alpha), anomaly_threshold)
                routes, _ = route(proba, in_set, anomaly, valid, pol)
                rates = route_rates(routes, outcomes)
                rows.append(
                    {
                        "alpha": float(alpha),
                        "t_quarantine": tq,
                        "t_rework": tr,
                        "cost": expected_cost(routes, outcomes, matrix),
                        "wrongful_scrap_rate": rates["wrongful_scrap_rate"],
                        "scrap_capture_rate": rates["scrap_capture_rate"],
                        "refer_share": rates["share_REFER"],
                    }
                )
    search = pd.DataFrame(rows)
    within_cap = search[search["wrongful_scrap_rate"] <= cap]
    feasible = within_cap[within_cap["scrap_capture_rate"] >= min_capture]
    search["feasible"] = search.index.isin(feasible.index)
    if len(feasible):
        best = feasible.loc[feasible["cost"].idxmin()]
    elif len(within_cap):
        # Capture target unreachable within the cap: catch as much scrap as possible.
        best = within_cap.sort_values(["scrap_capture_rate", "cost"], ascending=[False, True]).iloc[
            0
        ]
    else:
        best = search.sort_values(["t_quarantine", "cost"], ascending=[False, True]).iloc[0]
    policy = Policy(
        float(best["t_quarantine"]),
        float(best["t_rework"]),
        float(best["alpha"]),
        anomaly_threshold,
    )
    return policy, search
