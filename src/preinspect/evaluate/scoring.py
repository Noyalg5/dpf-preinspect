"""Scoring, calibration and seed aggregation of prediction tables."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preinspect.config import Config, data_dir
from preinspect.data.io import load_latent
from preinspect.evaluate.metrics import (
    OUTCOMES,
    expected_calibration_error,
    outcome_metrics,
    regression_metrics,
)
from preinspect.models.calibration import (
    TemperatureScaler,
    calibration_halves,
    make_calibrator,
    select_method,
)
from preinspect.models.tabular import save_model

PROBA = [f"p_{c}" for c in OUTCOMES]
LOGITS = [f"z_{c}" for c in OUTCOMES]


def with_truth(cfg: Config, preds: pd.DataFrame) -> pd.DataFrame:
    """Attach latent truth (evaluation only) and recorded labels to predictions."""
    latent = load_latent(data_dir(cfg))[
        ["unit_id", "outcome_true", "dp_ratio_post_clean_true", "integrity_true"]
    ]
    units = pd.read_parquet(data_dir(cfg) / "validation" / "units_clean.parquet")
    recorded = units[["unit_id", "disposition", "dp_ratio_post_clean"]]
    return preds.merge(latent, on="unit_id").merge(recorded, on="unit_id")


def calibrate(
    preds: pd.DataFrame, fit_ids: set[str], ccfg: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, dict, object]:
    """Calibrate probabilities on the calibration-fit units (recorded labels).

    Networks (tables with logits) use temperature scaling. For boosting
    models the method is either fixed in config or, with ``auto``, chosen
    by cross-validated log-loss on the calibration-fit units.
    """
    fit = preds[preds["unit_id"].isin(fit_ids)]
    y_idx = fit["disposition"].map(OUTCOMES.index).to_numpy()
    out = preds.copy()
    if all(c in preds for c in LOGITS):
        scaler = TemperatureScaler().fit(fit[LOGITS].to_numpy(), y_idx)
        out[PROBA] = scaler.transform(preds[LOGITS].to_numpy())
        return out, {"method": "temperature", "temperature": scaler.temperature}, scaler
    method, losses = ccfg["boosting"], {}
    if method == "auto":
        method, losses = select_method(
            fit[PROBA].to_numpy(), y_idx, ccfg["candidates"], ccfg["cv_folds"], seed
        )
    cal = make_calibrator(method).fit(fit[PROBA].to_numpy(), y_idx)
    out[PROBA] = cal.transform(preds[PROBA].to_numpy())
    return out, {"method": method, "cv_log_loss": losses}, cal


def method_comparison(
    preds: pd.DataFrame, fit_ids: set[str], ccfg: dict[str, Any], bins: int
) -> dict[str, float]:
    """Test-split ECE of every candidate calibration method (reported, not used to choose)."""
    fit = preds[preds["unit_id"].isin(fit_ids)]
    test = preds[preds["split"] == "test"]
    y_idx = fit["disposition"].map(OUTCOMES.index).to_numpy()
    out = {
        "uncalibrated": expected_calibration_error(
            test["outcome_true"].to_numpy(), test[PROBA].to_numpy(), bins
        )
    }
    for method in ccfg["candidates"]:
        cal = make_calibrator(method).fit(fit[PROBA].to_numpy(), y_idx)
        out[method] = expected_calibration_error(
            test["outcome_true"].to_numpy(), cal.transform(test[PROBA].to_numpy()), bins
        )
    return out


def score(df: pd.DataFrame, ece_bins: int) -> dict[str, Any]:
    """Headline metrics against the latent truth, plus recorded-label metrics."""
    proba = df[PROBA].to_numpy()
    truth = outcome_metrics(df["outcome_true"].to_numpy(), proba)
    recorded = outcome_metrics(df["disposition"].to_numpy(), proba)
    return {
        "truth": truth,
        "recorded": {k: recorded[k] for k in ("macro_f1", "balanced_accuracy", "recall_SCRAP")},
        "dp": regression_metrics(df["dp_ratio_post_clean_true"], df["dp_pred"]),
        "ece": expected_calibration_error(df["outcome_true"].to_numpy(), proba, ece_bins),
    }


def evaluate_predictions(cfg: Config, run_path: Path) -> dict[str, Any]:
    """Calibrate and score every prediction table in a run.

    Writes calibrated tables to ``predictions_calibrated/`` and returns
    per-model metrics on the test split, raw and calibrated.
    """
    units = pd.read_parquet(data_dir(cfg) / "validation" / "units_clean.parquet")
    fit_ids, conf_ids = calibration_halves(units.loc[units["split"] == "calib", "unit_id"].tolist())
    out_dir = run_path / "predictions_calibrated"
    out_dir.mkdir(exist_ok=True)
    results: dict[str, Any] = {
        "calibration_split": {"fit": len(fit_ids), "conformal": len(conf_ids)}
    }
    bins = cfg["evaluation"]["ece_bins"]
    for path in sorted((run_path / "predictions").glob("*.parquet")):
        preds = with_truth(cfg, pd.read_parquet(path))
        calibrated, info, calibrator = calibrate(preds, fit_ids, cfg["calibration"], cfg["seed"])
        if info["method"] != "temperature" or "cv_log_loss" in info:
            info["test_ece_by_method"] = method_comparison(preds, fit_ids, cfg["calibration"], bins)
        calibrated.to_parquet(out_dir / path.name, index=False)
        save_model(calibrator, run_path / "models" / "calibration" / f"{path.stem}.joblib")
        test_raw = preds[preds["split"] == "test"]
        test_cal = calibrated[calibrated["split"] == "test"]
        results[path.stem] = {
            "raw": score(test_raw, bins),
            "calibrated": score(test_cal, bins),
            "calibration": info,
        }
    (run_path / "calibration_units.json").write_text(
        pd.Series({"fit": sorted(fit_ids), "conformal": sorted(conf_ids)}).to_json()
    )
    return results


def seed_summary(results: dict[str, Any], name: str, key: str) -> dict[str, float]:
    """Mean and standard deviation of a metric across seeds ``<name>_s<k>``.

    ``key`` is a dotted path into the calibrated score, e.g. ``truth.macro_f1``.
    A model without seed suffixes is reported with ``n = 1``.
    """
    names = [k for k in results if re.fullmatch(rf"{re.escape(name)}(_s\d+)?", k)]
    vals = []
    for n in names:
        node: Any = results[n]["calibrated"]
        for part in key.split("."):
            node = node[part]
        vals.append(float(node))
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "n": int(arr.size),
    }
