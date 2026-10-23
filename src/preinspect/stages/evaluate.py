"""Stage ``evaluate``: score every model's predictions on the test split."""

from __future__ import annotations

import pandas as pd

from preinspect.config import Config, data_dir
from preinspect.data.io import load_latent
from preinspect.evaluate.metrics import (
    expected_calibration_error,
    outcome_metrics,
    regression_metrics,
)
from preinspect.models.tabular import PROBA_COLUMNS
from preinspect.runs import Run, open_run
from preinspect.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def with_truth(cfg: Config, preds: pd.DataFrame) -> pd.DataFrame:
    """Attach latent truth (evaluation only) and recorded labels to predictions."""
    latent = load_latent(data_dir(cfg))[["unit_id", "outcome_true", "dp_ratio_post_clean_true"]]
    units = pd.read_parquet(data_dir(cfg) / "validation" / "units_clean.parquet")
    recorded = units[["unit_id", "disposition", "dp_ratio_post_clean"]]
    return preds.merge(latent, on="unit_id").merge(recorded, on="unit_id")


def score(cfg: Config, preds: pd.DataFrame, split: str = "test") -> dict:
    """Headline metrics for one prediction table on one split."""
    df = with_truth(cfg, preds)
    df = df[df["split"] == split]
    proba = df[PROBA_COLUMNS].to_numpy()
    truth = outcome_metrics(df["outcome_true"].to_numpy(), proba)
    recorded = outcome_metrics(df["disposition"].to_numpy(), proba)
    return {
        "truth": truth,
        "recorded": {k: recorded[k] for k in ("macro_f1", "balanced_accuracy", "recall_SCRAP")},
        "dp": regression_metrics(df["dp_ratio_post_clean_true"], df["dp_pred"]),
        "ece": expected_calibration_error(
            df["outcome_true"].to_numpy(), proba, cfg["evaluation"]["ece_bins"]
        ),
    }


def run(cfg: Config, run_name: str | None = None) -> Run:
    """Evaluate every prediction table in the run."""
    run_ = open_run(cfg, run_name)
    setup_logging(log_file=run_.path / "logs" / "evaluate.log")
    results = {}
    for path in sorted((run_.path / "predictions").glob("*.parquet")):
        results[path.stem] = score(cfg, pd.read_parquet(path))
        t = results[path.stem]["truth"]
        log.info(
            "%-22s macro-F1 %.3f  SCRAP recall %.3f  dP MAE %.3f",
            path.stem,
            t["macro_f1"],
            t["recall_SCRAP"],
            results[path.stem]["dp"]["mae"],
        )
    run_.write_metrics("evaluate", {"models": results})
    return run_
