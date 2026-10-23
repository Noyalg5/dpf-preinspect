"""Split conformal prediction (Angelopoulos & Bates; Romano et al., 2020).

Outcome sets use Adaptive Prediction Sets (APS): a unit's set holds the most
probable classes, in order, while their cumulative probability stays within a
threshold learned on held-out calibration units. With exchangeable data the
set contains the true label with probability at least ``1 - alpha``.

ΔP-ratio intervals use absolute residuals: ``prediction ± q``, with ``q`` the
conformal quantile of held-out residuals.

The non-randomised APS score is used, which is slightly conservative but
gives the same set for the same input every time, a property operators
expect.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile of calibration scores."""
    n = len(scores)
    if n == 0:
        return float("inf")
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def aps_scores(proba: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """Cumulative probability mass down to and including the true class."""
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rank = np.argmax(order == y_idx[:, None], axis=1)
    return cum[np.arange(len(y_idx)), rank]


def aps_sets(proba: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean ``(n, K)`` membership: class ``y`` is in the set when its APS score <= qhat.

    The most probable class is always included, so a set is never empty.
    """
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    include_sorted = cum <= qhat + 1e-12
    include_sorted[:, 0] = True
    member = np.zeros_like(include_sorted)
    np.put_along_axis(member, order, include_sorted, axis=1)
    return member


class ConformalOutcome:
    """APS prediction sets at one or more miscoverage levels."""

    def __init__(self, alphas: list[float]) -> None:
        self.alphas = alphas
        self.qhat: dict[float, float] = {}

    def fit(self, proba: np.ndarray, y_idx: np.ndarray) -> ConformalOutcome:
        """Learn thresholds from calibration probabilities and labels."""
        scores = aps_scores(proba, y_idx)
        self.qhat = {a: conformal_quantile(scores, a) for a in self.alphas}
        return self

    def predict(self, proba: np.ndarray, alpha: float) -> np.ndarray:
        """Set membership for new units."""
        return aps_sets(proba, self.qhat[alpha])


class ConformalInterval:
    """Symmetric intervals for a regression target from absolute residuals."""

    def __init__(self, alphas: list[float]) -> None:
        self.alphas = alphas
        self.q: dict[float, float] = {}

    def fit(self, y: np.ndarray, y_pred: np.ndarray) -> ConformalInterval:
        """Learn half-widths from calibration residuals."""
        ok = np.isfinite(y) & np.isfinite(y_pred)
        resid = np.abs(y[ok] - y_pred[ok])
        self.q = {a: conformal_quantile(resid, a) for a in self.alphas}
        return self

    def predict(self, y_pred: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds."""
        return y_pred - self.q[alpha], y_pred + self.q[alpha]


def coverage(member: np.ndarray, y_idx: np.ndarray) -> dict[str, float]:
    """Empirical coverage and set-size statistics."""
    sizes = member.sum(axis=1)
    return {
        "coverage": float(member[np.arange(len(y_idx)), y_idx].mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_share": float((sizes == 1).mean()),
        "n": int(len(y_idx)),
    }
