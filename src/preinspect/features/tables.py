"""Per-unit feature tables assembled from every modality.

Column prefixes identify the modality group used in ablations:
``meta_`` (service metadata), ``flow_`` (incoming flow curve), ``cls_``
(classical image concepts), ``lrn_`` (learned image concepts), ``anom_``
(anomaly guard) and ``xmod_`` (cross-modal features that need both flow and
metadata). Classical features are cached next to the data, keyed by the
manifest, the classical config and the source code of the extractor.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from preinspect.classical.concepts import image_concepts
from preinspect.config import Config, data_dir
from preinspect.data.io import load_signals, read_image
from preinspect.features.flow import flow_features
from preinspect.features.metadata import metadata_features
from preinspect.utils.hashing import sha256_file, sha256_json
from preinspect.utils.logging import get_logger

log = get_logger(__name__)
GROUP_PREFIX = {
    "metadata": "meta_",
    "flow": "flow_",
    "classical": "cls_",
    "learned": "lrn_",
    "anomaly": "anom_",
}
VIEWS = ("inlet", "outlet")


def _classical_chunk(args: tuple) -> list[dict[str, Any]]:
    ddir, ccfg, rows = args
    out = []
    for uid, lighting, cpsi, diameter in rows:
        rec: dict[str, Any] = {"unit_id": uid}
        for view in VIEWS:
            concepts = image_concepts(
                read_image(Path(ddir), uid, view), lighting[view], cpsi, diameter, ccfg
            )
            rec.update({f"cls_{view}_{k}": v for k, v in concepts.items()})
        out.append(rec)
    return out


def _cache_key(cfg: Config) -> str:
    src = Path(__file__).parents[1] / "classical"
    sources = {p.name: sha256_file(p) for p in sorted(src.glob("*.py"))}
    manifest = sha256_file(data_dir(cfg) / "manifest.json")
    return sha256_json({"manifest": manifest, "classical": cfg["classical"], "src": sources})[:16]


def classical_table(cfg: Config, units: pd.DataFrame) -> pd.DataFrame:
    """Classical concepts for both views of every unit (cached)."""
    ddir = data_dir(cfg)
    cache = ddir / "cache" / f"classical_{_cache_key(cfg)}.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        if set(units["unit_id"]) <= set(cached["unit_id"]):
            return cached[cached["unit_id"].isin(units["unit_id"])]
    rows = [
        (
            r.unit_id,
            {"inlet": r.lighting_inlet, "outlet": r.lighting_outlet},
            float(r.cell_density_cpsi),
            float(r.diameter_mm),
        )
        for r in units.itertuples()
    ]
    workers = max(1, int(cfg["num_workers"]))
    chunks = [rows[i :: workers * 4] for i in range(workers * 4) if rows[i :: workers * 4]]
    args = [(str(ddir), cfg["classical"], c) for c in chunks]
    if workers == 1:
        records = [r for a in args for r in _classical_chunk(a)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = [r for part in pool.map(_classical_chunk, args) for r in part]
    table = pd.DataFrame(records)
    cache.parent.mkdir(exist_ok=True)
    table.to_parquet(cache, index=False)
    log.info("classical features for %d units cached at %s", len(table), cache.name)
    return table


def build_features(cfg: Config, units: pd.DataFrame) -> pd.DataFrame:
    """Join metadata, flow and classical features for the given units.

    Returns
    -------
    pandas.DataFrame
        One row per unit with ``unit_id``, ``split``, the recorded targets
        and every feature column.
    """
    signals = load_signals(data_dir(cfg))
    base = units[
        [
            "unit_id",
            "split",
            "disposition",
            "dp_ratio_post_clean",
            "integrity_pass",
            "lighting_inlet",
            "lighting_outlet",
        ]
    ].copy()
    meta = metadata_features(units, cfg)
    flow = flow_features(units, signals, cfg["generator"])
    table = pd.concat([base, meta, flow], axis=1)
    classical = classical_table(cfg, units)
    table = table.merge(classical, on="unit_id", how="left")
    for view in VIEWS:
        table[f"cls_{view}_lighting"] = pd.Categorical(
            table[f"lighting_{view}"], categories=list(cfg["generator"]["acquisition"]["lighting"])
        )
    return table.reset_index(drop=True)


def feature_columns(table: pd.DataFrame, groups: list[str]) -> list[str]:
    """Columns belonging to the requested modality groups.

    Cross-modal ``xmod_`` columns are included only when both flow and
    metadata are requested, since they need both.
    """
    prefixes = [GROUP_PREFIX[g] for g in groups]
    if "flow" in groups and "metadata" in groups:
        prefixes.append("xmod_")
    return [c for c in table.columns if any(c.startswith(p) for p in prefixes)]


def save_table(table: pd.DataFrame, path: Path) -> None:
    """Write a feature table and a column manifest next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    path.with_suffix(".columns.json").write_text(json.dumps(list(table.columns), indent=1))
