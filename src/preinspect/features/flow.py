"""Flow-curve features: fitted coefficients and normalised restriction.

The curve ``dP = a*Q + b*Q**2`` is fitted by least squares (no intercept)
on the points that survived sensor dropouts. The pressure drop at rated
flow is divided by the new-part value for the same geometry, which is a
datasheet figure per part number; here it is computed with the same
new-part relationship the simulator uses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from preinspect.synth.flow import new_part_dp_kpa


def fit_curve(q: np.ndarray, dp: np.ndarray) -> tuple[float, float, float, int]:
    """Least-squares fit of ``dp = a*q + b*q**2`` on finite points.

    Returns
    -------
    a, b : float
        Coefficients (NaN if fewer than three valid points).
    rms : float
        Root-mean-square residual in kPa.
    n_valid : int
        Number of points used.
    """
    ok = np.isfinite(dp)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan"), int(ok.sum())
    A = np.column_stack([q[ok], q[ok] ** 2])
    (a, b), *_ = np.linalg.lstsq(A, dp[ok], rcond=None)
    rms = float(np.sqrt(np.mean((A @ np.array([a, b]) - dp[ok]) ** 2)))
    return float(a), float(b), rms, int(ok.sum())


def flow_features(units: pd.DataFrame, signals: np.ndarray, gcfg: dict[str, Any]) -> pd.DataFrame:
    """Per-unit flow features.

    Parameters
    ----------
    units : pandas.DataFrame
        Accepted units (needs ``signal_index`` and geometry columns).
    signals : numpy.ndarray
        All flow curves, shape ``(N, 2, n_points)``.
    gcfg : dict
        Generator configuration (rated flow and new-part relationship).

    Returns
    -------
    pandas.DataFrame
        Indexed like ``units``: ``flow_a``, ``flow_b``, ``flow_dp_rated_kpa``,
        ``flow_dp_ratio``, ``flow_fit_rms``, ``flow_dropouts``.
    """
    q_rated = gcfg["flow"]["rated_q_m3s"]
    rows = []
    for _, u in units.iterrows():
        q, dp = signals[int(u["signal_index"])]
        a, b, rms, n_valid = fit_curve(q.astype(float), dp.astype(float))
        dp_rated = a * q_rated + b * q_rated**2
        rows.append(
            {
                "flow_a": a,
                "flow_b": b,
                "flow_dp_rated_kpa": dp_rated,
                "flow_dp_ratio": dp_rated / new_part_dp_kpa(gcfg, u.to_dict()),
                "flow_fit_rms": rms,
                "flow_dropouts": float(q.size - n_valid),
            }
        )
    return pd.DataFrame(rows, index=units.index)


class LoadResidual:
    """Flow restriction relative to what the unit's weight gain predicts.

    A loaded filter should restrict flow in proportion to its deposit mass;
    restriction *lower* than expected for the weight gain points to a leak
    path (crack, missing plugs) that lets gas bypass the walls. The line is
    fitted on training units only.
    """

    def __init__(self) -> None:
        self.coef: tuple[float, float] = (0.0, 1.0)

    def fit(self, load_g_per_l: pd.Series, dp_ratio: pd.Series) -> LoadResidual:
        """Fit ``dp_ratio ~ slope * load + intercept`` on finite pairs."""
        ok = load_g_per_l.notna() & dp_ratio.notna()
        slope, intercept = np.polyfit(load_g_per_l[ok], dp_ratio[ok], 1)
        self.coef = (float(slope), float(intercept))
        return self

    def transform(self, load_g_per_l: pd.Series, dp_ratio: pd.Series) -> pd.Series:
        """Observed minus expected restriction."""
        slope, intercept = self.coef
        return dp_ratio - (slope * load_g_per_l + intercept)
