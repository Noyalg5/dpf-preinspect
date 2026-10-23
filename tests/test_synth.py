"""Synthetic generator: determinism, mask/label consistency, physics directions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from preinspect.config import data_dir
from preinspect.data.io import decode_bitmask, load_latent, load_scenes, read_mask
from preinspect.synth.flow import flow_curve
from preinspect.synth.generate import generate_dataset
from preinspect.synth.population import sample_condition, sample_metadata


def test_same_seed_gives_identical_files(tiny_data, tiny_cfg) -> None:
    generate_dataset(tiny_cfg)
    first = json.loads((data_dir(tiny_data) / "manifest.json").read_text())["files"]
    second = json.loads((data_dir(tiny_cfg) / "manifest.json").read_text())["files"]
    assert first == second


def test_masks_match_scene_defects(tiny_data) -> None:
    ddir = data_dir(tiny_data)
    classes = tiny_data["defect_classes"]
    scenes = load_scenes(ddir)
    expected = found = 0
    for uid, rec in scenes.items():
        for view in ("inlet", "outlet"):
            present = decode_bitmask(read_mask(ddir, uid, view), len(classes)).any(axis=(1, 2))
            in_scene = {d["cls"] for d in rec["scene"]["defects"] if d["view"] == view}
            drawn = {c for c, p in zip(classes, present, strict=True) if p}
            assert drawn <= in_scene, (uid, view, drawn - in_scene)
            expected += len(in_scene)
            found += len(drawn)
    # Tiny defects can vanish under a clamp or at 128 px; nearly all must survive.
    assert found >= 0.9 * expected


def test_outcome_labels_follow_latent_truth(tiny_data) -> None:
    latent = load_latent(data_dir(tiny_data))
    assert (latent.loc[~latent["integrity_true"], "outcome_true"] == "SCRAP").all()
    assert set(latent.loc[latent["melt_severity"] > 0, "outcome_true"]) <= {"SCRAP"}
    units = pd.read_parquet(data_dir(tiny_data) / "units.parquet")
    novel = units.loc[units["split"] == "novel", "unit_id"]
    assert (latent.set_index("unit_id").loc[novel, "melt_severity"] > 0).all()
    assert set(units.loc[units["split"] == "train", "unit_id"]).isdisjoint(novel)


def test_splits_are_chronological(tiny_data) -> None:
    units = pd.read_parquet(data_dir(tiny_data) / "units.parquet").drop_duplicates("unit_id")
    dates = units.groupby("split")["arrival_date"].agg(["min", "max"])
    assert dates.loc["train", "max"] <= dates.loc["calib", "min"]
    assert dates.loc["calib", "max"] <= dates.loc["test", "min"]
    assert dates.loc["test", "max"] <= dates.loc["shift", "min"]


def test_leaks_lower_pressure_drop(tiny_cfg) -> None:
    g = tiny_cfg["generator"]
    meta, rng = sample_metadata(g, 1, "U1", 10, "main")
    cond = sample_condition(g, meta, rng)
    cond.update(bypass_fraction=0.0)
    healthy = flow_curve(g, meta, cond, np.random.default_rng(0))[1]
    cond.update(bypass_fraction=0.3)
    leaking = flow_curve(g, meta, cond, np.random.default_rng(0))[1]
    ok = np.isfinite(healthy) & np.isfinite(leaking)
    assert (leaking[ok] < healthy[ok]).all()
