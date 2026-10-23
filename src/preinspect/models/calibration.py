"""Post-hoc probability calibration (Guo et al., 2017).

- Temperature scaling for networks: one scalar ``T`` divides the logits,
  chosen to minimise negative log-likelihood on held-out data. It never
  changes the arg-max class.
- One-vs-rest sigmoid (Platt) or isotonic maps for boosting models, then
  renormalised so each row sums to one.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class TemperatureScaler:
    """Single-temperature calibration of logits."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_idx: np.ndarray) -> TemperatureScaler:
        """Choose ``T`` in [0.05, 20] minimising NLL."""

        def nll(log_t: float) -> float:
            z = logits / np.exp(log_t)
            return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y_idx)), y_idx]))

        res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
        self.temperature = float(np.exp(res.x))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Calibrated probabilities."""
        return softmax(logits / self.temperature, axis=1)


class OneVsRestCalibrator:
    """Per-class sigmoid or isotonic calibration of probabilities."""

    def __init__(self, method: str = "sigmoid") -> None:
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"unknown calibration method {method}")
        self.method = method
        self.maps: list = []

    def fit(self, proba: np.ndarray, y_idx: np.ndarray) -> OneVsRestCalibrator:
        """Fit one map per class on held-out probabilities.

        A class absent from (or making up all of) the calibration labels gets
        a constant map, since no monotone map can be fitted to one label value.
        """
        self.maps = []
        for k in range(proba.shape[1]):
            target = (y_idx == k).astype(int)
            if target.min() == target.max():
                self.maps.append(float(target.mean()))
            elif self.method == "isotonic":
                m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                self.maps.append(m.fit(proba[:, k], target))
            else:
                m = LogisticRegression()
                self.maps.append(m.fit(_logit(proba[:, k]).reshape(-1, 1), target))
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Calibrated, renormalised probabilities."""
        cols = []
        for k, m in enumerate(self.maps):
            if isinstance(m, float):
                cols.append(np.full(len(proba), m))
            elif self.method == "isotonic":
                cols.append(m.predict(proba[:, k]))
            else:
                cols.append(m.predict_proba(_logit(proba[:, k]).reshape(-1, 1))[:, 1])
        out = np.clip(np.column_stack(cols), EPS, None)
        return out / out.sum(axis=1, keepdims=True)


class ProbabilityTemperature:
    """Temperature scaling applied to log-probabilities of any classifier.

    Log-probabilities are logits up to a per-row constant, which softmax
    ignores, so a single temperature can be fitted exactly as for a network.
    One parameter makes it the most stable choice for small calibration sets.
    """

    def __init__(self) -> None:
        self.scaler = TemperatureScaler()

    def fit(self, proba: np.ndarray, y_idx: np.ndarray) -> ProbabilityTemperature:
        """Fit the temperature on held-out probabilities."""
        self.scaler.fit(np.log(np.clip(proba, EPS, 1.0)), y_idx)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Calibrated probabilities."""
        return self.scaler.transform(np.log(np.clip(proba, EPS, 1.0)))


def make_calibrator(method: str):
    """Probability calibrator by name: ``sigmoid``, ``isotonic`` or ``temperature``."""
    if method == "temperature":
        return ProbabilityTemperature()
    return OneVsRestCalibrator(method)


def select_method(
    proba: np.ndarray, y_idx: np.ndarray, methods: list[str], folds: int, seed: int
) -> tuple[str, dict[str, float]]:
    """Choose the calibration method with the lowest cross-validated log-loss.

    Uses only the calibration-fit units, so the choice never sees test data.
    """
    from sklearn.model_selection import StratifiedKFold

    k = max(2, min(folds, int(np.bincount(y_idx).min())))
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    losses = {}
    for method in methods:
        nll = []
        for train, held in splitter.split(proba, y_idx):
            cal = make_calibrator(method).fit(proba[train], y_idx[train])
            p = np.clip(cal.transform(proba[held]), EPS, 1.0)
            nll.append(float(-np.mean(np.log(p[np.arange(len(held)), y_idx[held]]))))
        losses[method] = float(np.mean(nll))
    return min(losses, key=losses.get), losses


def calibration_halves(unit_ids: list[str]) -> tuple[set[str], set[str]]:
    """Split calibration units into a calibration-fit half and a conformal half.

    Alternating assignment by sorted unit ID keeps both halves spread over
    the calibration period.
    """
    ordered = sorted(unit_ids)
    return set(ordered[0::2]), set(ordered[1::2])
