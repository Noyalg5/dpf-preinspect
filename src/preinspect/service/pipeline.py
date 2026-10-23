"""End-to-end inspection of a unit from raw inputs to a routed decision.

Used by the API, the validation campaign and the latency benchmark, so every
number in the validation report comes from the code path the station runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from preinspect.decision.conformal import aps_sets
from preinspect.decision.rationale import rationale
from preinspect.decision.routing import Policy, route
from preinspect.features.flow import curve_features
from preinspect.features.metadata import metadata_features
from preinspect.models.patchcore import PatchCore
from preinspect.models.tabular import OUTCOMES, PROBA_COLUMNS, load_model
from preinspect.models.unet import UNet
from preinspect.models.vision import describe_arrays
from preinspect.runs import Run
from preinspect.service.image_quality import check_images
from preinspect.service.input_checks import check_unit
from preinspect.utils.device import resolve_device

VIEWS = ("inlet", "outlet")


@dataclass
class UnitInput:
    """Raw inputs from the pre-inspection station for one unit."""

    unit_id: str
    images: dict[str, np.ndarray]
    curve: np.ndarray
    meta: dict[str, Any]
    lighting: dict[str, str]
    extra: dict[str, Any] = field(default_factory=dict)


class InspectionPipeline:
    """Loaded models plus the deployed decision policy."""

    def __init__(
        self,
        cfg: dict[str, Any],
        run: Run,
        device: torch.device | None = None,
        folds: list[int] | None = None,
    ) -> None:
        self.cfg, self.run = cfg, run
        self.device = device or resolve_device(cfg["device"])
        models = run.path / "models"
        k = cfg["perception"]["folds"]
        use = folds if folds is not None else list(range(k))
        n_cls = len(cfg["perception"]["classes"])
        self.unets = []
        for f in use:
            net = UNet(n_cls, pretrained=False)
            net.load_state_dict(
                torch.load(models / "perception" / f"unet_fold{f}.pt", map_location="cpu")
            )
            self.unets.append(net.to(self.device).eval())
        self.banks = [
            PatchCore(cfg["anomaly"], self.device, cfg["seed"] + f).load_state(
                torch.load(models / "anomaly" / f"bank_fold{f}.pt", map_location="cpu")
            )
            for f in use
        ]
        name = cfg["deploy"]["model"]
        self.model = load_model(models / "tabular" / f"{name}.joblib")
        self.calibrator = load_model(models / "calibration" / f"{name}.joblib")
        pol = json.loads((models / "policy.json").read_text())
        self.policy = Policy(
            pol["t_quarantine"], pol["t_rework"], pol["alpha"], pol["anomaly_threshold"]
        )
        self.qhat = {float(a): q for a, q in pol["qhat"].items()}
        self.interval_q = {float(a): q for a, q in pol["interval_q"].items()}
        self.residual = json.loads((models / "load_residual.json").read_text())
        quality_path = models / "image_quality.json"
        self.quality = json.loads(quality_path.read_text()) if quality_path.exists() else None
        pca_path = models / "embedding_pca.joblib"
        self.pca = load_model(pca_path) if pca_path.exists() else None
        prov = json.loads((run.path / "provenance.json").read_text())
        self.versions = {
            "model_version": run.name,
            "data_version": prov["data_manifest_sha256"],
            "code_version": prov["git"]["commit"],
            "deployed_model": name,
        }

    def _rows(self, units: list[UnitInput], described: list[dict]) -> pd.DataFrame:
        meta = pd.DataFrame([u.meta for u in units])
        table = metadata_features(meta, self.cfg).reset_index(drop=True)
        flow = pd.DataFrame([curve_features(self.cfg["generator"], u.meta, u.curve) for u in units])
        table = pd.concat([table, flow], axis=1)
        table["xmod_load_residual"] = table["flow_dp_ratio"] - (
            self.residual["slope"] * table["meta_load_g_per_l"] + self.residual["intercept"]
        )
        for i, _ in enumerate(units):
            for v, view in enumerate(VIEWS):
                rec = described[2 * i + v]
                for c, val in rec["lrn"].items():
                    table.loc[i, f"lrn_{view}_{c}"] = val
                for c, val in rec["anom"].items():
                    table.loc[i, f"anom_{view}_{c}"] = val
        for c in ("max", "unexplained"):
            table[f"anom_{c}"] = table[[f"anom_{v}_{c}" for v in VIEWS]].max(axis=1)
        for col in self.model.columns:
            if col not in table:
                table[col] = np.nan
        return table

    def inspect_batch(self, units: list[UnitInput], return_maps: bool = False) -> list[dict]:
        """Inspect several units; returns one result dictionary per unit."""
        cfg = self.cfg
        issues = [check_unit(u.meta, u.images, u.curve, u.lighting, cfg) for u in units]
        if self.quality is not None:
            for found, u in zip(issues, units, strict=True):
                if not found:
                    found += check_images(
                        u.images,
                        u.lighting,
                        float(u.meta.get("diameter_mm") or 0),
                        self.quality,
                        cfg,
                    )
        size = cfg["generator"]["image_size"]
        images, lighting, diameters = [], [], []
        for u in units:
            for view in VIEWS:
                img = u.images.get(view)
                ok = img is not None and img.ndim == 3 and img.shape[:2] == (size, size)
                images.append(img if ok else np.zeros((size, size, 3), np.uint8))
                lighting.append(u.lighting.get(view, "diffuse"))
                diameters.append(float(u.meta.get("diameter_mm") or 267.0))
        described = []
        batch = cfg["perception"]["infer_batch"]
        for s in range(0, len(images), batch):
            described += describe_arrays(
                cfg,
                images[s : s + batch],
                lighting[s : s + batch],
                diameters[s : s + batch],
                self.unets,
                self.banks,
                self.device,
            )
        table = self._rows(units, described)
        raw = self.model.predict(table.assign(unit_id=[u.unit_id for u in units], split="live"))
        proba = self.calibrator.transform(raw[PROBA_COLUMNS].to_numpy())
        member = aps_sets(proba, self.qhat[self.policy.alpha])
        valid = np.array([not i for i in issues])
        routes, reasons = route(
            proba, member[:, 2], table["anom_unexplained"].to_numpy(), valid, self.policy
        )
        interval_alpha = self.cfg["routing"]["interval_alpha"]
        q = self.interval_q[interval_alpha]
        results = []
        for i, u in enumerate(units):
            p = dict(zip(OUTCOMES, proba[i].round(4).tolist(), strict=True))
            pset = [c for c, m in zip(OUTCOMES, member[i], strict=True) if m]
            feats = table.iloc[i].to_dict()
            res = {
                "unit_id": u.unit_id,
                "valid": bool(valid[i]),
                "input_issues": issues[i],
                "probabilities": p,
                "prediction_set": pset,
                "dp_ratio": {
                    "prediction": float(raw["dp_pred"].iloc[i]),
                    "lower": float(raw["dp_pred"].iloc[i] - q),
                    "upper": float(raw["dp_pred"].iloc[i] + q),
                    "coverage": 1 - interval_alpha,
                },
                "anomaly": {
                    "score": float(feats["anom_max"]),
                    "unexplained": float(feats["anom_unexplained"]),
                    "threshold": self.policy.anomaly_threshold,
                },
                "route": str(routes[i]),
                "reason": str(reasons[i]),
                "rationale": rationale(
                    feats, p, pset, str(routes[i]), str(reasons[i]), cfg["rationale"]
                ),
                "defects": {view: described[2 * i + v]["lrn"] for v, view in enumerate(VIEWS)},
                "features": {
                    k: (None if pd.isna(v) else v)
                    for k, v in feats.items()
                    if isinstance(v, int | float | np.floating)
                },
                **self.versions,
            }
            res["monitoring"] = self._monitoring(u, described[2 * i : 2 * i + 2])
            if return_maps:
                res["maps"] = {
                    view: {
                        "prob": described[2 * i + v]["prob"],
                        "anomaly": described[2 * i + v]["amap"],
                    }
                    for v, view in enumerate(VIEWS)
                }
            results.append(res)
        return results

    def _monitoring(self, unit: UnitInput, views: list[dict]) -> dict[str, Any]:
        """Snapshot for drift monitoring: numeric metadata and the image-embedding projection."""
        from preinspect.features.metadata import NUMERIC

        meta = {
            c: (
                None
                if unit.meta.get(c) is None or pd.isna(unit.meta.get(c))
                else float(unit.meta[c])
            )
            for c in NUMERIC
        }
        emb = None
        if self.pca is not None and all(v.get("embedding") is not None for v in views):
            vec = np.mean([v["embedding"] for v in views], axis=0)[None, :]
            emb = [round(float(x), 5) for x in self.pca.transform(vec)[0]]
        return {"metadata": meta, "embedding": emb}

    def inspect(self, unit: UnitInput, return_maps: bool = True) -> dict:
        """Inspect one unit."""
        return self.inspect_batch([unit], return_maps)[0]


def unit_from_dataset(
    ddir: Path, row: pd.Series, signals: np.ndarray, images: dict[str, np.ndarray] | None = None
) -> UnitInput:
    """Build a :class:`UnitInput` from a dataset record (optionally with replacement images)."""
    from preinspect.data.io import read_image
    from preinspect.features.metadata import NUMERIC

    imgs = images or {v: read_image(ddir, row["unit_id"], v) for v in VIEWS}
    meta = {c: row[c] for c in [*NUMERIC, "material", "application"]}
    meta = {
        k: (float(v) if isinstance(v, int | float | np.integer | np.floating) else str(v))
        for k, v in meta.items()
    }
    return UnitInput(
        row["unit_id"],
        imgs,
        signals[int(row["signal_index"])],
        meta,
        {v: row[f"lighting_{v}"] for v in VIEWS},
    )
