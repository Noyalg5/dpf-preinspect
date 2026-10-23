"""FastAPI service: inspection, operator decisions, audit, monitoring.

Start with ``preinspect api --config demo``. The app loads the latest run of
the profile (or ``PREINSPECT_RUN``) and appends every call to the audit trail.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from preinspect.config import Config, data_dir, load_config, project_root
from preinspect.runs import Run, open_run
from preinspect.service import monitoring as mon
from preinspect.service.audit import AuditTrail
from preinspect.service.mes import MockMes
from preinspect.service.overlays import overlays
from preinspect.service.pipeline import InspectionPipeline, UnitInput, unit_from_dataset
from preinspect.service.schemas import (
    DISCLAIMER,
    DecisionRequest,
    InspectRequest,
    InspectResponse,
    ResultRequest,
)
from preinspect.utils.hashing import canonical_json, sha256_bytes

VIEWS = ("inlet", "outlet")


def _path(value: str, env: str) -> Path:
    path = Path(os.environ.get(env, value))
    return path if path.is_absolute() else project_root() / path


def _decode(data: bytes, view: str) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{view} image could not be decoded; send a JPEG or PNG file")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _errors(exc: ValidationError) -> list[dict[str, str]]:
    return [
        {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()
    ]


def create_app(cfg: Config | None = None, run: Run | None = None) -> FastAPI:
    """Build the application around one trained run."""
    cfg = cfg or load_config(os.environ.get("PREINSPECT_CONFIG", "demo"))
    run = run or open_run(cfg, os.environ.get("PREINSPECT_RUN"))
    pipe = InspectionPipeline(cfg, run)
    audit = AuditTrail(_path(cfg["service"]["audit_db"], "PREINSPECT_AUDIT_DB"))
    mes = MockMes(_path(cfg["service"]["mes_db"], "PREINSPECT_MES_DB"))
    features = pd.read_parquet(run.path / "features" / "features.parquet")
    app = FastAPI(title="DPF pre-inspection service", version="0.1.0", description=DISCLAIMER)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_version": pipe.versions["model_version"],
            "device": str(pipe.device),
        }

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        g = cfg["generator"]
        return {
            **pipe.versions,
            "generator_version": g["version"],
            "disclaimer": DISCLAIMER,
            "policy": pipe.policy.as_dict(),
            "defect_classes": cfg["perception"]["classes"],
            "inputs": {
                "image_size_px": g["image_size"],
                "lighting_modes": list(g["acquisition"]["lighting"]),
                "flow_rates_m3s": g["flow"]["q_m3s"],
                "materials": list(g["substrate"]["materials"]),
                "applications": list(g["applications"]),
            },
        }

    @app.post("/inspect", response_model=InspectResponse)
    async def inspect(
        inlet: UploadFile = File(...), outlet: UploadFile = File(...), request: str = Form(...)
    ) -> dict[str, Any]:
        try:
            req = InspectRequest.model_validate_json(request)
        except ValidationError as exc:
            raise HTTPException(422, _errors(exc)) from exc
        raw = {"inlet": await inlet.read(), "outlet": await outlet.read()}
        images = {v: _decode(raw[v], v) for v in VIEWS}
        curve = np.array(
            [req.flow.q_m3s, [np.nan if x is None else x for x in req.flow.dp_kpa]], dtype=float
        )
        unit = UnitInput(req.unit_id, images, curve, req.metadata.model_dump(), req.lighting)
        result = pipe.inspect(unit, return_maps=req.include_maps)
        input_hash = sha256_bytes(
            raw["inlet"]
            + raw["outlet"]
            + canonical_json(req.model_dump(exclude={"include_maps"})).encode()
        )
        result["inspection_id"] = audit.log_inspection(result, input_hash)
        mes.publish_route(req.unit_id, result["route"], result["inspection_id"], "system")
        if req.include_maps:
            result["maps"] = overlays(
                images,
                result.pop("maps"),
                cfg["perception"]["mask_threshold"],
                pipe.policy.anomaly_threshold,
            )
        return result

    @app.post("/decision")
    def decision(req: DecisionRequest) -> dict[str, Any]:
        record = audit.log_operator(
            req.inspection_id, req.unit_id, req.route, req.accepted, req.reason, req.operator
        )
        mes.publish_route(req.unit_id, req.route, req.inspection_id, "operator")
        return {"record_id": record, "route": req.route}

    @app.post("/result")
    def result(req: ResultRequest) -> dict[str, Any]:
        return {
            "record_id": audit.log_result(req.unit_id, req.disposition, req.dp_ratio_post_clean)
        }

    @app.get("/audit/verify")
    def audit_verify() -> dict[str, Any]:
        rep = audit.verify()
        return {"ok": rep.ok, "records": rep.n_records, "message": rep.message}

    @app.get("/records")
    def records(limit: int = 1000) -> list[dict[str, Any]]:
        df = audit.records(limit)
        return df.drop(columns=["prev_hash"]).to_dict("records")

    @app.get("/mes/queue")
    def mes_queue() -> list[dict[str, Any]]:
        return mes.queue().to_dict("records")

    @app.get("/monitoring")
    def monitoring(window: int = 200) -> dict[str, Any]:
        return monitoring_summary(cfg, audit, features, window)

    @app.get("/samples")
    def samples(limit: int = 50) -> list[str]:
        units = pd.read_parquet(data_dir(cfg) / "validation" / "units_clean.parquet")
        return units.loc[units["split"] == "test", "unit_id"].head(limit).tolist()

    @app.get("/samples/{unit_id}")
    def sample(unit_id: str) -> dict[str, Any]:
        return sample_payload(cfg, unit_id)

    return app


def monitoring_summary(
    cfg: Config, audit: AuditTrail, features: pd.DataFrame, window: int
) -> dict[str, Any]:
    """Throughput, route mix, p-chart, drift and calibration over recent inspections."""
    m = cfg["monitoring"]
    recs = audit.records()
    insp = recs[recs["record_type"] == "inspection"].tail(window)
    if insp.empty:
        return {"inspections": 0}
    payload = pd.json_normalize(insp["payload"].tolist())
    routes = payload["route"]
    ref = features[features["split"] == "train"]
    meta_ref = ref.filter(like="meta_").rename(columns=lambda c: c[5:])
    meta_cur = pd.DataFrame([p.get("monitoring", {}).get("metadata", {}) for p in insp["payload"]])
    emb_cur = pd.DataFrame(
        [p.get("monitoring", {}).get("embedding") or [] for p in insp["payload"]]
    )
    emb_ref = ref.filter(like="emb_")
    emb_cur.columns = list(emb_ref.columns)[: emb_cur.shape[1]]
    results = recs[recs["record_type"] == "downstream_result"]
    res = pd.DataFrame(
        {
            "unit_id": results["unit_id"],
            "disposition": [p["disposition"] for p in results["payload"]],
        }
    )
    times = pd.to_datetime(insp["created_utc"])
    span_h = max((times.max() - times.min()).total_seconds() / 3600, 1e-9)
    return {
        "inspections": int(len(insp)),
        "throughput_per_hour": float(len(insp) / span_h) if len(insp) > 1 else None,
        "route_mix": routes.value_counts().to_dict(),
        "p_chart": mon.p_chart(routes, m["batch_size"]).to_dict("records"),
        "metadata_drift": mon.drift_table(
            meta_ref, meta_cur, m["psi_bins"], m["psi_warn"], m["psi_alert"]
        ),
        "embedding_drift": mon.drift_table(
            emb_ref, emb_cur, m["psi_bins"], m["psi_warn"], m["psi_alert"]
        ),
        "calibration": mon.calibration_over_time(
            pd.DataFrame(
                {
                    "unit_id": insp["unit_id"],
                    "probabilities": payload.filter(like="probabilities.")
                    .rename(columns=lambda c: c.split(".")[1])
                    .to_dict("records"),
                }
            ),
            res,
            m["batch_size"],
        ).to_dict("records"),
        "operator_overrides": int(
            sum(
                1
                for p in recs.loc[recs["record_type"] == "operator_decision", "payload"]
                if not p["accepted"]
            )
        ),
    }


def sample_payload(cfg: Config, unit_id: str) -> dict[str, Any]:
    """A dataset unit packaged as an API request (demo convenience)."""
    from preinspect.data.io import load_signals

    ddir = data_dir(cfg)
    units = pd.read_parquet(ddir / "validation" / "units_clean.parquet").set_index("unit_id")
    if unit_id not in units.index:
        raise HTTPException(404, f"unit {unit_id} is not in the accepted dataset")
    row = units.loc[unit_id].copy()
    row["unit_id"] = unit_id
    unit = unit_from_dataset(ddir, row, load_signals(ddir))
    jpg = {
        v: base64.b64encode(
            cv2.imencode(".jpg", cv2.cvtColor(unit.images[v], cv2.COLOR_RGB2BGR))[1].tobytes()
        ).decode()
        for v in VIEWS
    }
    meta = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in unit.meta.items()}
    for k in ("active_regen_count", "previous_cleans"):
        meta[k] = int(meta[k])
    return {
        "images_jpeg_base64": jpg,
        "request": {
            "unit_id": unit_id,
            "metadata": meta,
            "lighting": unit.lighting,
            "flow": {
                "q_m3s": unit.curve[0].tolist(),
                "dp_kpa": [None if np.isnan(x) else float(x) for x in unit.curve[1]],
            },
        },
        "recorded": {
            "disposition": row["disposition"],
            "dp_ratio_post_clean": float(row["dp_ratio_post_clean"]),
        },
    }
