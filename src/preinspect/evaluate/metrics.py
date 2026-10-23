"""Metrics used throughout: classification, regression and calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

OUTCOMES = ["PASS", "REWORK", "SCRAP"]


def outcome_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray | None = None) -> dict:
    """Macro-F1, balanced accuracy, per-class recall/precision and one-vs-rest AUC.

    Parameters
    ----------
    y_true : numpy.ndarray of str
        True outcomes.
    proba : numpy.ndarray
        ``(n, 3)`` probabilities in ``OUTCOMES`` order.
    pred : numpy.ndarray of str, optional
        Decisions; defaults to the arg-max class.
    """
    y_true = np.asarray(y_true)
    pred = np.asarray(OUTCOMES)[proba.argmax(axis=1)] if pred is None else np.asarray(pred)
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "macro_f1": float(
            f1_score(y_true, pred, labels=OUTCOMES, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(np.mean(y_true == pred)),
    }
    recalls = recall_score(y_true, pred, labels=OUTCOMES, average=None, zero_division=0)
    precisions = precision_score(y_true, pred, labels=OUTCOMES, average=None, zero_division=0)
    aucs = []
    for k, cls in enumerate(OUTCOMES):
        out[f"recall_{cls}"] = float(recalls[k])
        out[f"precision_{cls}"] = float(precisions[k])
        positives = y_true == cls
        auc = (
            float(roc_auc_score(positives, proba[:, k]))
            if 0 < positives.sum() < len(y_true)
            else None
        )
        out[f"auc_{cls}"] = auc
        if auc is not None:
            aucs.append(auc)
    out["auc_macro"] = float(np.mean(aucs)) if aucs else None
    out["confusion"] = confusion_matrix(y_true, pred, labels=OUTCOMES).tolist()
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Mean absolute error and RMSE of the ΔP-ratio prediction."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true[ok], y_pred[ok])),
        "rmse": float(np.sqrt(np.mean((y_true[ok] - y_pred[ok]) ** 2))),
    }


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width bins (Guo et al., 2017)."""
    y_idx = np.array([OUTCOMES.index(y) for y in y_true])
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def reliability_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Per-bin confidence, accuracy and count for reliability diagrams."""
    y_idx = np.array([OUTCOMES.index(y) for y in y_true])
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_idx
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    return [
        {
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": int((bins == b).sum()),
            "confidence": float(conf[bins == b].mean()) if (bins == b).any() else None,
            "accuracy": float(correct[bins == b].mean()) if (bins == b).any() else None,
        }
        for b in range(n_bins)
    ]
