"""Tests of explanations against what the simulator knows to be true.

- Energy-in-mask and pointing game: does a heatmap concentrate on the real
  defect pixels, compared with the share of the image the defect covers?
- Deletion and insertion (Petsiuk et al., 2018): removing the most salient
  pixels first should lower the class score faster than a random order.
- Model randomisation (Adebayo et al., 2018): heatmaps from a network whose
  top layers are re-initialised should look different; if they do not, the
  method is not explaining the model.
- Driver agreement: global attributions grouped by physical concept should
  rank the same concepts highly as the simulator's true drivers do.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


def energy_in_mask(cam: np.ndarray, mask: np.ndarray) -> float:
    """Share of heatmap mass that falls inside the defect mask."""
    total = float(cam.sum())
    return float((cam * mask).sum() / total) if total > 0 else float("nan")


def pointing_hit(cam: np.ndarray, mask: np.ndarray, tolerance_px: int) -> bool:
    """Whether the heatmap's peak lies within ``tolerance_px`` of the mask."""
    k = 2 * tolerance_px + 1
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return bool(grown[y, x])


def tolerance_area(mask: np.ndarray, tolerance_px: int) -> float:
    """Chance level of the pointing game: area share of the dilated mask."""
    k = 2 * tolerance_px + 1
    return float((cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0).mean())


def deletion_insertion(
    score: Callable[[np.ndarray], float],
    image: np.ndarray,
    blurred: np.ndarray,
    saliency: np.ndarray,
    steps: list[float],
) -> tuple[list[float], list[float]]:
    """Class score as salient pixels are removed (deletion) or restored (insertion).

    ``image`` and ``blurred`` are ``(H, W, 3)``; removed pixels take the
    blurred image's value, which removes detail without adding new edges.
    """
    order = np.argsort(-saliency.ravel())
    flat_img, flat_blur = image.reshape(-1, 3), blurred.reshape(-1, 3)
    deletion, insertion = [], []
    for frac in steps:
        k = int(round(frac * order.size))
        deleted = flat_img.copy()
        deleted[order[:k]] = flat_blur[order[:k]]
        inserted = flat_blur.copy()
        inserted[order[:k]] = flat_img[order[:k]]
        deletion.append(score(deleted.reshape(image.shape)))
        insertion.append(score(inserted.reshape(image.shape)))
    return deletion, insertion


def curve_auc(values: list[float], steps: list[float]) -> float:
    """Area under a curve sampled at ``steps`` (trapezoidal rule)."""
    return float(np.trapezoid(values, steps))


def map_similarity(a: np.ndarray, b: np.ndarray, stride: int = 4) -> float:
    """Spearman rank correlation between two heatmaps (subsampled); 0 if either is constant."""
    x, y = a[::stride, ::stride].ravel(), b[::stride, ::stride].ravel()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    rho = spearmanr(x, y).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def group_of(feature: str, groups: dict[str, Any]) -> str | None:
    """Physical-concept group of a feature name (first matching substring)."""
    for name, spec in groups.items():
        if any(token in feature for token in spec["features"]):
            return name
    return None


def model_group_importance(mean_abs: pd.Series, groups: dict[str, Any]) -> dict[str, float]:
    """Sum of mean absolute attributions per concept group."""
    out = {g: 0.0 for g in groups}
    for feature, value in mean_abs.items():
        g = group_of(str(feature), groups)
        if g is not None:
            out[g] += float(value)
    return out


def latent_group_importance(
    latent: pd.DataFrame, train: np.ndarray, test: np.ndarray, groups: dict[str, Any], seed: int
) -> dict[str, float]:
    """True driver importance: permutation importance of a model on latent variables."""
    cols = sorted({c for spec in groups.values() for c in spec["latent"]})
    X = latent[cols].astype(float)
    y = (latent["outcome_true"] == "SCRAP").astype(int)
    model = HistGradientBoostingClassifier(random_state=seed).fit(X[train], y[train])
    imp = permutation_importance(
        model, X[test], y[test], n_repeats=5, random_state=seed, scoring="roc_auc"
    )
    per_col = dict(zip(cols, imp.importances_mean, strict=True))
    return {
        g: float(sum(max(0.0, per_col[c]) for c in spec["latent"])) for g, spec in groups.items()
    }


def rank_agreement(model_imp: dict[str, float], true_imp: dict[str, float]) -> float:
    """Spearman correlation between model and true concept-group rankings."""
    keys = list(true_imp)
    x, y = [model_imp[k] for k in keys], [true_imp[k] for k in keys]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rho = spearmanr(x, y).statistic
    return float(rho) if np.isfinite(rho) else float("nan")
