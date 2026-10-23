"""Stage ``train``: fit every model into a new run folder.

Steps (each writes into the run folder):
1. feature tables (metadata, flow, classical concepts, cross-modal);
2. gradient-boosting models for each modality combination;
3. deep models (perception, anomaly guard, fusion) once added.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from preinspect.config import Config
from preinspect.data.validate import require_valid
from preinspect.features.flow import LoadResidual
from preinspect.features.tables import build_features, feature_columns, save_table
from preinspect.models.tabular import fit_tabular, save_model
from preinspect.runs import Run, create_run
from preinspect.utils.logging import get_logger, setup_logging
from preinspect.utils.seeding import seed_everything
from preinspect.utils.timing import timed

log = get_logger(__name__)


def prepare_features(cfg: Config, units: pd.DataFrame, run: Run) -> pd.DataFrame:
    """Build the feature table and add the cross-modal load residual."""
    table = build_features(cfg, units)
    train = (table["split"] == "train").to_numpy()
    residual = LoadResidual().fit(
        table.loc[train, "meta_load_g_per_l"], table.loc[train, "flow_dp_ratio"]
    )
    table["xmod_load_residual"] = residual.transform(
        table["meta_load_g_per_l"], table["flow_dp_ratio"]
    )
    run.write_json(
        "models/load_residual.json", {"slope": residual.coef[0], "intercept": residual.coef[1]}
    )
    save_table(table, run.sub("features", "features.parquet"))
    return table


def fit_tabular_models(
    cfg: Config,
    table: pd.DataFrame,
    run: Run,
    names: list[str] | None = None,
    models: dict[str, list[str]] | None = None,
) -> None:
    """Fit one boosting model per modality combination and store predictions."""
    train = (table["split"] == "train").to_numpy()
    specs = models or cfg["tabular"]["models"]
    for name, groups in specs.items():
        if names is not None and name not in names:
            continue
        cols = feature_columns(table, groups)
        model = fit_tabular(name, table, cols, train, cfg["tabular"], cfg["seed"])
        save_model(model, run.sub("models", "tabular", f"{name}.joblib"))
        model.predict(table).to_parquet(run.sub("predictions", f"{name}.parquet"), index=False)
        log.info("fitted %s on %d features", name, len(cols))


def run(cfg: Config) -> Run:
    """Train all models for the profile and return the run handle."""
    units = require_valid(cfg)
    run_ = create_run(cfg)
    setup_logging(log_file=run_.path / "logs" / "train.log")
    seed_everything(cfg["seed"])
    timings: dict[str, float] = {}
    with timed(timings, "features"):
        table = prepare_features(cfg, units, run_)
    with timed(timings, "tabular_models"):
        fit_tabular_models(cfg, table, run_)
    counts = table["split"].value_counts().to_dict()
    run_.write_metrics("train", {"timings_s": timings, "units_by_split": counts})
    (run_.path / "train_done.json").write_text(json.dumps({"ok": True}))
    log.info("run folder: %s", run_.path)
    return run_


def split_mask(table: pd.DataFrame, split: str) -> np.ndarray:
    """Boolean mask of rows in a split."""
    return (table["split"] == split).to_numpy()
