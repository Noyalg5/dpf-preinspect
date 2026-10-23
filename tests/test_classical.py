"""Classical image processing measured against rendered faces with known geometry."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from preinspect.classical.concepts import image_concepts
from preinspect.classical.lattice import estimate_lattice, expected_pitch_px
from preinspect.classical.localise import localise_substrate
from preinspect.config import load_config
from preinspect.synth.acquisition import sample_acquisition
from preinspect.synth.render import render_view

SIZE = 256


@pytest.fixture(scope="module")
def cfg():
    return load_config("demo")


def _scene(defects: list[dict]) -> dict:
    return {
        "unit_id": "T",
        "material": "cordierite",
        "diameter_mm": 267.0,
        "n_cells": 34,
        "segment_cells": 0,
        "soot_film": {"inlet": 0.2, "outlet": 0.02},
        "texture_seed": 7,
        "defects": defects,
        "nuisances": [],
    }


def _render(cfg, defects, lighting="diffuse", rotation=20.0):
    g = cfg["generator"]
    acq = sample_acquisition(
        g,
        np.random.default_rng(3),
        "cam_a",
        lighting=lighting,
        clamp=None,
        reflection=None,
        rotation_deg=rotation,
    )
    image, _, info = render_view(g, cfg["defect_classes"], _scene(defects), "inlet", acq, SIZE)
    return image, info


def _concepts(cfg, image, lighting="diffuse"):
    return image_concepts(image, lighting, 200.0, 267.0, cfg["classical"])


def test_localisation_and_lattice_match_truth(cfg) -> None:
    image, info = _render(cfg, [])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    loc = localise_substrate(gray, cfg["classical"], "diffuse")
    assert math.hypot(loc.cx - info["cx"], loc.cy - info["cy"]) < 3.0
    assert abs(loc.radius / info["radius_px"] - 1.0) < 0.05
    lat = estimate_lattice(
        gray, loc, cfg["classical"], expected_pitch_px(cfg["classical"], 200.0, 267.0, loc.radius)
    )
    assert abs(lat.pitch / (2 * info["radius_px"] / 34) - 1.0) < 0.05
    err = (lat.angle - math.radians(info["rotation_deg"])) % (math.pi / 2)
    assert min(err, math.pi / 2 - err) < math.radians(2.0)


def test_crack_lengthens_measured_ridges(cfg) -> None:
    crack = {
        "cls": "crack",
        "view": "inlet",
        "subtype": "radial",
        "severity": 0.6,
        "width": 0.03,
        "points": [[0.1 * k, 0.05 * k] for k in range(1, 10)],
    }
    clean = _concepts(cfg, _render(cfg, [])[0])
    cracked = _concepts(cfg, _render(cfg, [crack])[0])
    assert cracked["crack_length_mm"] > clean["crack_length_mm"] + 50.0


def test_missing_plugs_counted_under_backlight(cfg) -> None:
    cells = [[i, j] for i in range(10, 24, 2) for j in (10, 20)]
    defect = {"cls": "missing_plug", "view": "inlet", "cells": cells}
    clean = _concepts(cfg, _render(cfg, [], "backlight")[0], "backlight")
    leaky = _concepts(cfg, _render(cfg, [defect], "backlight")[0], "backlight")
    assert clean["missing_plug_count"] <= 1
    assert leaky["missing_plug_count"] >= 0.7 * len(cells)


def test_chip_raises_perimeter_deviation(cfg) -> None:
    chip = {
        "cls": "chip",
        "view": "inlet",
        "severity": 0.8,
        "angle": 1.0,
        "half_width": 0.45,
        "depth": 0.13,
        "harm": [],
    }
    clean = _concepts(cfg, _render(cfg, [])[0])
    chipped = _concepts(cfg, _render(cfg, [chip])[0])
    assert chipped["chip_depth_max"] > clean["chip_depth_max"] + 0.02
