"""Re-imaging and re-measuring units, as a repeat visit to the station would.

A unit's scene is fixed; only the acquisition changes: it is re-seated
(offset, rotation), exposure drifts slightly and sensor noise is fresh. The
flow bench re-measures the same latent condition with new noise and the
station's own calibration bias.
"""

from __future__ import annotations

import copy
from typing import Any

import cv2
import numpy as np

from preinspect.synth.flow import flow_curve
from preinspect.synth.render import render_view

COND_FIELDS = ("soot_load_gL", "ash_load_gL", "face_plug_extent", "glazing", "bypass_fraction")


def jittered_acquisition(
    acq: dict[str, Any], rng: np.random.Generator, jitter: dict[str, float], **overrides: Any
) -> dict[str, Any]:
    """Copy of an acquisition record with re-seating, exposure drift and new noise."""
    new = copy.deepcopy(acq)
    new["offset"] = [o + float(rng.normal(0, jitter["offset_sd"])) for o in new["offset"]]
    new["rotation_deg"] = new["rotation_deg"] + float(rng.uniform(-1, 1) * jitter["rotation_deg"])
    new["exposure_ev"] = new["exposure_ev"] + float(rng.normal(0, jitter["exposure_sd"]))
    new["seed"] = int(rng.integers(0, 2**31))
    new.update(overrides)
    return new


def rerender(
    cfg: dict[str, Any],
    record: dict[str, Any],
    view: str,
    rng: np.random.Generator,
    jitter: dict[str, float],
    size: int | None = None,
    **overrides: Any,
) -> np.ndarray:
    """Render a view again with a jittered acquisition; optionally at another resolution.

    Rendering at a lower ``size`` and upsampling back imitates a lower-resolution
    camera while keeping the input size the models expect.
    """
    g = cfg["generator"]
    out = g["image_size"]
    acq = jittered_acquisition(record["acquisition"][view], rng, jitter, **overrides)
    image, _, _ = render_view(g, cfg["defect_classes"], record["scene"], view, acq, size or out)
    if size and size != out:
        image = cv2.resize(image, (out, out), interpolation=cv2.INTER_CUBIC)
    return image


def remeasure_flow(
    cfg: dict[str, Any],
    meta: dict[str, Any],
    latent: dict[str, Any],
    rng: np.random.Generator,
    bias: float = 0.0,
) -> np.ndarray:
    """New flow-bench curve for the same unit with a station calibration bias."""
    cond = {k: float(latent[k]) for k in COND_FIELDS}
    curve = flow_curve(cfg["generator"], meta, cond, rng)
    curve[1] = curve[1] * (1.0 + bias)
    return curve
