"""Dataset statistics for the data card and the gallery figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preinspect.config import Config, data_dir
from preinspect.data.io import decode_bitmask, load_latent, load_scenes, read_image, read_mask
from preinspect.report.style import SERIES, save_figure

VIEWS = ("inlet", "outlet")


def mask_presence(ddir: Path, unit_ids: list[str], classes: list[str]) -> pd.DataFrame:
    """Image-level presence of each defect class, one row per (unit, view)."""
    rows = []
    for uid in unit_ids:
        for view in VIEWS:
            present = decode_bitmask(read_mask(ddir, uid, view), len(classes)).any(axis=(1, 2))
            rows.append({"unit_id": uid, "view": view, **dict(zip(classes, present, strict=True))})
    return pd.DataFrame(rows)


def dataset_stats(cfg: Config) -> dict[str, Any]:
    """Compute the counts the data card is rendered from and write ``stats.json``."""
    ddir = data_dir(cfg)
    manifest = json.loads((ddir / "manifest.json").read_text())
    status = json.loads((ddir / "validation" / "status.json").read_text())
    clean = pd.read_parquet(ddir / "validation" / "units_clean.parquet")
    latent = load_latent(ddir).set_index("unit_id")
    scenes = load_scenes(ddir)
    classes = cfg["defect_classes"]
    presence = mask_presence(ddir, clean["unit_id"].tolist(), classes)
    unit_presence = presence.groupby("unit_id")[classes].any()
    clean = clean.join(unit_presence, on="unit_id")
    truth = latent.loc[clean["unit_id"], "outcome_true"].to_numpy()

    def crosstab(col: str) -> dict[str, dict[str, int]]:
        return pd.crosstab(clean["split"], clean[col]).to_dict("index")

    nuisance = {"dust": 0, "marker": 0, "reflection": 0, "clamp": 0}
    for rec in scenes.values():
        for n in rec["scene"]["nuisances"]:
            nuisance[n["kind"]] += 1
        for view in VIEWS:
            nuisance["reflection"] += rec["acquisition"][view]["reflection"] is not None
            nuisance["clamp"] += rec["acquisition"][view]["clamp"] is not None
    sizes = {
        k: sum((ddir / f).stat().st_size for f in manifest["files"] if f.startswith(k))
        for k in ("images", "masks")
    }
    stats = {
        "profile": cfg["name"],
        "generator_version": manifest["generator_version"],
        "seed": manifest["seed"],
        "image_size": cfg["generator"]["image_size"],
        "n_units": manifest["counts"]["units"],
        "n_records": manifest["counts"]["records"],
        "n_files": len(manifest["files"]),
        "bytes": sizes,
        "validation": status,
        "faults_injected": manifest["counts"]["faults"],
        "disposition_by_split": crosstab("disposition"),
        "material_by_split": crosstab("material"),
        "cpsi_by_split": crosstab("cell_density_cpsi"),
        "application_by_split": crosstab("application"),
        "lighting_inlet": clean["lighting_inlet"].value_counts().to_dict(),
        "camera_by_split": crosstab("camera_profile"),
        "defect_units_by_split": clean.groupby("split")[classes].sum().astype(int).to_dict("index"),
        "defect_free_share": float((~clean[classes].any(axis=1)).mean()),
        "image_defect_share": presence[classes].mean().to_dict(),
        "nuisance_counts": nuisance,
        "max_regen_temp_missing_share": float(clean["max_regen_temp_c"].isna().mean()),
        "label_vs_truth_disagreement": float((clean["disposition"].to_numpy() != truth).mean()),
        "msa_units": int(clean["msa"].sum()),
        "inspectors": {k: v["bias"] for k, v in cfg["generator"]["inspectors"].items()},
        "generation": json.loads((ddir / "logs" / "generation.json").read_text()),
    }
    (ddir / "stats.json").write_text(json.dumps(stats, indent=1, default=int))
    return stats


def gallery(cfg: Config, path: Path) -> Path:
    """Grid of example faces, one per defect class, with mask contours."""
    ddir = data_dir(cfg)
    classes = cfg["defect_classes"]
    clean = pd.read_parquet(ddir / "validation" / "units_clean.parquet")
    scenes = load_scenes(ddir)
    panels: list[tuple[str, str, str, int | None]] = []
    for k, cls in enumerate(classes):
        for uid in clean["unit_id"]:
            views = [d["view"] for d in scenes[uid]["scene"]["defects"] if d["cls"] == cls]
            if views and scenes[uid]["acquisition"][views[0]]["lighting"] != "backlight":
                panels.append((cls.replace("_", " "), uid, views[0], k))
                break
    for uid in clean["unit_id"]:
        sc = scenes[uid]
        if not sc["scene"]["defects"] and sc["scene"]["nuisances"]:
            panels.append(("nuisances only", uid, sc["scene"]["nuisances"][0]["view"], None))
            break
    for uid in clean["unit_id"]:
        sc = scenes[uid]
        leaks = [d["view"] for d in sc["scene"]["defects"] if d["cls"] in ("crack", "missing_plug")]
        if leaks and sc["acquisition"][leaks[0]]["lighting"] == "backlight":
            panels.append(("backlight leak", uid, leaks[0], None))
            break
    cols = 5
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.5 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (title, uid, view, k) in zip(np.ravel(axes), panels, strict=False):
        img = read_image(ddir, uid, view)
        if k is not None:
            m = decode_bitmask(read_mask(ddir, uid, view), len(classes))[k].astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            colour = tuple(int(SERIES[k][i : i + 2], 16) for i in (1, 3, 5))
            img = cv2.drawContours(img.copy(), contours, -1, colour, 1)
        ax.imshow(img)
        ax.set_title(f"{title}\n{uid} {view}", fontsize=8)
    fig.suptitle(
        "Synthetic face images with ground-truth mask outlines (synthetic data)", fontsize=10
    )
    return save_figure(fig, path)
