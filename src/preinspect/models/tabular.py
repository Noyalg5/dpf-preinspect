"""Gradient-boosting models (scikit-learn HistGradientBoosting) on feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

OUTCOMES = ["PASS", "REWORK", "SCRAP"]
PROBA_COLUMNS = [f"p_{c}" for c in OUTCOMES]


@dataclass
class TabularModel:
    """Outcome classifier and ΔP-ratio regressor sharing one feature set."""

    name: str
    columns: list[str]
    classifier: HistGradientBoostingClassifier
    regressor: HistGradientBoostingRegressor

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        """Class probabilities (columns ``p_PASS`` ...) and ΔP-ratio predictions."""
        X = table[self.columns]
        proba = self.classifier.predict_proba(X)
        order = [list(self.classifier.classes_).index(c) for c in OUTCOMES]
        out = table[["unit_id", "split"]].copy()
        out[PROBA_COLUMNS] = proba[:, order]
        out["dp_pred"] = self.regressor.predict(X)
        return out.reset_index(drop=True)


def fit_tabular(
    name: str,
    table: pd.DataFrame,
    columns: list[str],
    train: np.ndarray,
    tcfg: dict[str, Any],
    seed: int,
) -> TabularModel:
    """Fit both heads on the training rows.

    Parameters
    ----------
    name : str
        Model name (e.g. ``classical_flow_meta``).
    table : pandas.DataFrame
        Feature table with ``disposition`` and ``dp_ratio_post_clean`` targets.
    columns : list of str
        Feature columns to use.
    train : numpy.ndarray of bool
        Training-row mask.
    tcfg : dict
        ``tabular`` configuration section.
    seed : int
        Random state.
    """
    # A column with no observed value in training carries no information (and
    # breaks histogram binning), so it is dropped; the model keeps its own list.
    columns = [c for c in columns if table.loc[train, c].notna().any()]
    X, rows = table.loc[train, columns], table.loc[train]
    clf = HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=seed, **tcfg["classifier"]
    ).fit(X, rows["disposition"])
    reg = HistGradientBoostingRegressor(
        categorical_features="from_dtype", random_state=seed, **tcfg["regressor"]
    ).fit(X, rows["dp_ratio_post_clean"])
    return TabularModel(name, columns, clf, reg)


def save_model(model: Any, path: Path) -> None:
    """Persist a fitted model with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    """Load a model saved with :func:`save_model`."""
    return joblib.load(path)
