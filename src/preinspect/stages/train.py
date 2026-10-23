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
import torch

from preinspect.config import Config, data_dir
from preinspect.data.validate import require_valid
from preinspect.features.flow import LoadResidual
from preinspect.features.tables import build_features, feature_columns, save_table
from preinspect.models.fusion_train import predict_fusion, train_fusion
from preinspect.models.tabular import fit_tabular, save_model
from preinspect.models.vision import fit_vision_models, vision_pass
from preinspect.runs import Run, create_run
from preinspect.utils.device import resolve_device
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


def add_vision_features(
    cfg: Config, table: pd.DataFrame, run: Run, device: torch.device
) -> tuple[pd.DataFrame, dict]:
    """Train cross-fitted perception and anomaly models; add their concepts to the table."""
    ddir = data_dir(cfg)
    unets, banks, info = fit_vision_models(cfg, ddir, table, run, device)
    vision, metrics, embeddings = vision_pass(cfg, ddir, table, unets, banks, device)
    table = table.merge(vision, on="unit_id", how="left")
    table = add_embedding_reference(cfg, table, embeddings, run)
    save_table(table, run.sub("features", "features.parquet"))
    run.write_json("metrics/vision.json", metrics)
    return table, {"vision_models": info}


def add_embedding_reference(
    cfg: Config, table: pd.DataFrame, embeddings: dict, run: Run
) -> pd.DataFrame:
    """Project image embeddings onto principal components fitted on training units.

    The ``emb_`` columns are not model inputs; they are the reference
    distribution for embedding-drift monitoring in service.
    """
    from sklearn.decomposition import PCA

    train_ids = [u for u in table.loc[table["split"] == "train", "unit_id"] if u in embeddings]
    k = min(cfg["monitoring"]["embedding_components"], len(train_ids))
    pca = PCA(n_components=k, random_state=cfg["seed"]).fit(
        np.stack([embeddings[u] for u in train_ids])
    )
    save_model(pca, run.sub("models", "embedding_pca.joblib"))
    ids = [u for u in table["unit_id"] if u in embeddings]
    proj = pca.transform(np.stack([embeddings[u] for u in ids]))
    cols = [f"emb_{i + 1}" for i in range(k)]
    return table.merge(
        pd.DataFrame(proj, columns=cols).assign(unit_id=ids), on="unit_id", how="left"
    )


def fit_quality_reference(cfg: Config, table: pd.DataFrame, run: Run) -> None:
    """Acceptable image-quality ranges from a sample of training images."""
    from preinspect.data.io import read_image
    from preinspect.service.image_quality import fit_reference, reference_path

    train = table[table["split"] == "train"]
    n = min(cfg["quality"]["train_images"] // 2, len(train))
    sample = train.sample(n, random_state=cfg["seed"])
    images = [
        (
            read_image(data_dir(cfg), r.unit_id, v),
            getattr(r, f"lighting_{v}"),
            float(r.meta_diameter_mm),
        )
        for r in sample.itertuples()
        for v in ("inlet", "outlet")
    ]
    reference_path(run.path).write_text(json.dumps(fit_reference(images, cfg), indent=1))


def fit_fusion_models(cfg: Config, units: pd.DataFrame, run: Run, device: torch.device) -> dict:
    """Train each end-to-end variant for the configured number of seeds."""
    ddir = data_dir(cfg)
    info = {}
    for name, modalities in cfg["fusion"]["variants"].items():
        for s in range(cfg["fusion"]["seeds"]):
            model, prep, history = train_fusion(
                cfg, ddir, units, modalities, cfg["seed"] + s, device
            )
            torch.save(
                {"state": model.state_dict(), "modalities": modalities, "prep": prep},
                run.sub("models", "fusion", f"{name}_s{s}.pt"),
            )
            preds = predict_fusion(model, prep, cfg, ddir, units, device)
            preds.to_parquet(run.sub("predictions", f"{name}_s{s}.parquet"), index=False)
            info[f"{name}_s{s}"] = history
    return info


def run(cfg: Config) -> Run:
    """Train all models for the profile and return the run handle."""
    units = require_valid(cfg)
    run_ = create_run(cfg)
    setup_logging(log_file=run_.path / "logs" / "train.log")
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    timings: dict[str, float] = {}
    with timed(timings, "features"):
        table = prepare_features(cfg, units, run_)
    with timed(timings, "vision_models"):
        table, vision_info = add_vision_features(cfg, table, run_, device)
    with timed(timings, "quality_reference"):
        fit_quality_reference(cfg, table, run_)
    with timed(timings, "tabular_models"):
        fit_tabular_models(cfg, table, run_)
    with timed(timings, "fusion_models"):
        fusion_info = fit_fusion_models(cfg, units, run_, device)
    counts = table["split"].value_counts().to_dict()
    run_.write_metrics(
        "train",
        {
            "timings_s": timings,
            "units_by_split": counts,
            "device": str(device),
            **vision_info,
            "fusion": fusion_info,
        },
    )
    (run_.path / "train_done.json").write_text(json.dumps({"ok": True}))
    log.info("run folder: %s", run_.path)
    return run_


def split_mask(table: pd.DataFrame, split: str) -> np.ndarray:
    """Boolean mask of rows in a split."""
    return (table["split"] == split).to_numpy()
