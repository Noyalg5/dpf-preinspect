"""Generate a complete synthetic dataset (the virtual inspection cell).

Output layout under ``data/<profile>/``::

    images/<unit_id>_<view>.jpg   face images
    masks/<unit_id>_<view>.png    per-class defect bitmasks
    units.parquet                 recorded metadata and test results (faults included)
    signals.npy                   incoming flow-bench curves, shape (N, 2, n_points)
    latent.parquet                latent truth, never a model input
    faults.parquet                ground truth of injected data-quality faults
    inspectors.parquet            simulated inspector calls for the MSA study
    scenes.jsonl                  scene and acquisition records (for re-imaging)
    manifest.json                 config, seed, generator version, SHA-256 of every file
"""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from preinspect.config import Config, data_dir
from preinspect.synth.acquisition import sample_acquisition
from preinspect.synth.faults import inject_faults
from preinspect.synth.flow import flow_curve
from preinspect.synth.inspectors import simulate_inspectors
from preinspect.synth.outcome import downstream_outcome
from preinspect.synth.population import sample_condition, sample_metadata
from preinspect.synth.render import render_view
from preinspect.synth.scene import sample_scene
from preinspect.utils.hashing import sha256_file, sha256_json
from preinspect.utils.logging import get_logger
from preinspect.utils.seeding import make_rng

log = get_logger(__name__)
VIEWS = ("inlet", "outlet")
# Folders and files that are not part of the hashed dataset.
UNHASHED_DIRS = {"validation", "figures", "logs"}
UNHASHED_FILES = {"manifest.json", "stats.json"}


def arrival_plan(gcfg: dict[str, Any], seed: int) -> list[tuple[str, int]]:
    """Arrival day for every unit, sorted chronologically.

    Novel-defect units arrive during the test period; the domain-shift line
    starts after the main period ends.
    """
    rng = make_rng(seed, "arrivals")
    arr, n = gcfg["arrival"], gcfg["n_units"]
    days = arr["main_days"]
    test_start = days * (gcfg["splits"]["train"] + gcfg["splits"]["calib"])
    shift_start = days + arr["shift_gap_days"]
    plan = [("main", int(d)) for d in rng.uniform(0, days, n["main"])]
    plan += [("novel", int(d)) for d in rng.uniform(test_start, days, n["novel"])]
    plan += [
        ("shift", int(d))
        for d in rng.uniform(shift_start, shift_start + arr["shift_days"], n["shift"])
    ]
    return sorted(plan, key=lambda p: p[1])


def build_unit(cfg: Config, unit_id: str, day: int, population: str) -> dict[str, Any]:
    """Sample everything about one unit except its pixels."""
    g, seed = cfg["generator"], cfg["seed"]
    meta, rng = sample_metadata(g, seed, unit_id, day, population)
    cond = sample_condition(g, meta, rng)
    outcome = downstream_outcome(g, cond, rng)
    curve = flow_curve(g, meta, cond, rng)
    scene = sample_scene(g, meta, cond, seed)
    camera = g["acquisition"]["shift_camera" if population == "shift" else "main_camera"]
    acq = {v: sample_acquisition(g, make_rng(seed, "acq", unit_id, v, 0), camera) for v in VIEWS}
    cracks = cond.pop("cracks")
    latent = {
        "unit_id": unit_id,
        **cond,
        "crack_count": len(cracks),
        "crack_max_severity": max((c["severity"] for c in cracks), default=0.0),
        "crack_types": ",".join(sorted({c["type"] for c in cracks})),
        **{k: v for k, v in outcome.items() if k.endswith("_true") or k.startswith("residual")},
    }
    record = {
        **meta,
        **{
            k: v
            for k, v in outcome.items()
            if not (k.endswith("_true") or k.startswith("residual"))
        },
        "camera_profile": camera,
        **{f"lighting_{v}": acq[v]["lighting"] for v in VIEWS},
    }
    return {"units": record, "latent": latent, "curve": curve, "scene": scene, "acq": acq}


def assign_splits(units: pd.DataFrame, gcfg: dict[str, Any]) -> pd.Series:
    """Chronological train/calib/test split of the main population."""
    split = units["population"].copy()
    main = units.index[units["population"] == "main"]
    n_train = int(round(len(main) * gcfg["splits"]["train"]))
    n_calib = int(round(len(main) * gcfg["splits"]["calib"]))
    split.loc[main[:n_train]] = "train"
    split.loc[main[n_train : n_train + n_calib]] = "calib"
    split.loc[main[n_train + n_calib :]] = "test"
    return split


def select_msa_units(units: pd.DataFrame, excluded: set[str], n: int, seed: int) -> list[str]:
    """Pick about ``n`` fault-free test units, balanced across dispositions."""
    rng = make_rng(seed, "msa")
    pool = units[(units["split"] == "test") & ~units["unit_id"].isin(excluded)]
    by_class = {
        k: list(rng.permutation(g["unit_id"].to_numpy())) for k, g in pool.groupby("disposition")
    }
    chosen: list[str] = []
    while len(chosen) < n and any(by_class.values()):
        for ids in by_class.values():
            if ids and len(chosen) < n:
                chosen.append(str(ids.pop()))
    return sorted(chosen)


def _render_chunk(args: tuple) -> list[dict[str, Any]]:
    gcfg, classes, size, out_dir, items = args
    out_dir = Path(out_dir)
    infos = []
    for unit_id, scene, acq in items:
        for view in VIEWS:
            image, mask, info = render_view(gcfg, classes, scene, view, acq[view], size)
            quality = gcfg["acquisition"]["cameras"][acq[view]["camera"]]["jpeg_quality"]
            ok, jpg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
            )
            (out_dir / "images" / f"{unit_id}_{view}.jpg").write_bytes(jpg.tobytes())
            ok, png = cv2.imencode(".png", mask)
            (out_dir / "masks" / f"{unit_id}_{view}.png").write_bytes(png.tobytes())
            infos.append({"unit_id": unit_id, "view": view, **info})
    return infos


def render_all(cfg: Config, items: list[tuple], out_dir: Path) -> list[dict[str, Any]]:
    """Render every unit's two views, in parallel when ``num_workers > 1``."""
    g = cfg["generator"]
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    workers = max(1, int(cfg["num_workers"]))
    n_chunks = workers * 4
    chunks = [items[i::n_chunks] for i in range(n_chunks) if items[i::n_chunks]]
    args = [(g, cfg["defect_classes"], g["image_size"], str(out_dir), c) for c in chunks]
    if workers == 1:
        return [info for a in args for info in _render_chunk(a)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [info for part in pool.map(_render_chunk, args) for info in part]


def write_manifest(cfg: Config, out_dir: Path, counts: dict[str, Any]) -> dict[str, Any]:
    """Hash every data file and write ``manifest.json``."""
    files = sorted(
        p
        for p in out_dir.rglob("*")
        if p.is_file()
        and p.name not in UNHASHED_FILES
        and not UNHASHED_DIRS & set(p.relative_to(out_dir).parts)
    )
    manifest = {
        "generator_version": cfg["generator"]["version"],
        "profile": cfg["name"],
        "seed": cfg["seed"],
        "generator_config_sha256": sha256_json(cfg["generator"]),
        "generator_config": cfg["generator"],
        "defect_classes": cfg["defect_classes"],
        "counts": counts,
        "files": {str(p.relative_to(out_dir)): sha256_file(p) for p in files},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def generate_dataset(cfg: Config) -> Path:
    """Generate the dataset for a profile, replacing any previous copy.

    Parameters
    ----------
    cfg : dict
        Resolved profile configuration.

    Returns
    -------
    Path
        Dataset folder.
    """
    g, seed = cfg["generator"], cfg["seed"]
    out_dir = data_dir(cfg)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    t0 = time.perf_counter()
    plan = arrival_plan(g, seed)
    records = [build_unit(cfg, f"DPF-{k + 1:06d}", day, pop) for k, (pop, day) in enumerate(plan)]
    units = pd.DataFrame([r["units"] for r in records])
    units["split"] = assign_splits(units, g)
    units["signal_index"] = np.arange(len(units))
    latent = pd.DataFrame([r["latent"] for r in records])
    signals = np.stack([r["curve"] for r in records])
    log.info("sampled %d units in %.1f s", len(units), time.perf_counter() - t0)

    faulty, faults = inject_faults(units, g, seed)
    msa_ids = select_msa_units(units, set(faults["unit_id"]), g["msa_units"], seed)
    inspectors = simulate_inspectors(msa_ids, units, latent, g, seed)
    faulty["msa"] = faulty["unit_id"].isin(msa_ids)

    t1 = time.perf_counter()
    items = [(r["units"]["unit_id"], r["scene"], r["acq"]) for r in records]
    infos = render_all(cfg, items, out_dir)
    log.info("rendered %d images in %.1f s", len(infos), time.perf_counter() - t1)
    by_unit: dict[str, dict[str, Any]] = {}
    for info in infos:
        by_unit.setdefault(info.pop("unit_id"), {})[info.pop("view")] = info

    faulty.to_parquet(out_dir / "units.parquet", index=False)
    latent.to_parquet(out_dir / "latent.parquet", index=False)
    faults.to_parquet(out_dir / "faults.parquet", index=False)
    inspectors.to_parquet(out_dir / "inspectors.parquet", index=False)
    np.save(out_dir / "signals.npy", signals)
    with open(out_dir / "scenes.jsonl", "w", encoding="utf-8") as handle:
        for r in records:
            uid = r["units"]["unit_id"]
            handle.write(
                json.dumps(
                    {
                        "unit_id": uid,
                        "scene": r["scene"],
                        "acquisition": r["acq"],
                        "render": by_unit[uid],
                    }
                )
                + "\n"
            )
    counts = {
        "units": int(len(units)),
        "records": int(len(faulty)),
        "splits": units["split"].value_counts().to_dict(),
        "faults": faults["fault_type"].value_counts().to_dict(),
        "msa_units": len(msa_ids),
    }
    write_manifest(cfg, out_dir, counts)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "logs" / "generation.json").write_text(
        json.dumps(
            {
                "seconds_total": round(time.perf_counter() - t0, 1),
                "seconds_render": round(time.perf_counter() - t1, 1),
                "workers": cfg["num_workers"],
            },
            indent=1,
        )
    )
    log.info("dataset written to %s in %.1f s", out_dir, time.perf_counter() - t0)
    return out_dir
