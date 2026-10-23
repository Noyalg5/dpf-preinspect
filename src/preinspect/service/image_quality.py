"""Image-quality gate: is this image like the ones the models were trained on?

Per image, after locating the substrate:

- face brightness (median grey level on the face);
- sharpness (variance of the Laplacian on the face);
- out-of-range area: share of face pixels darker or brighter than almost
  anything seen on training faces (an obstruction, glare or clipping);
- seating: substrate centre offset and radius relative to the expected
  position for the part's diameter and the station's field of view.

Acceptable ranges are percentiles of the same statistics over training
images, widened by a margin. Nothing here is fitted to corrupted images; an
image outside the training envelope simply is not trusted and the unit is
referred to an inspector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from preinspect.classical.localise import localise_substrate

STATS = ("brightness", "sharpness", "dark_area", "bright_area", "offset", "radius_ratio")
LABELS = {
    "brightness": "exposure outside the trained range",
    "sharpness": "image sharpness outside the trained range (focus or motion blur)",
    "dark_area": "part of the face is obstructed or unlit",
    "bright_area": "glare or clipping on the face",
    "offset": "unit not seated centrally in the fixture",
    "radius_ratio": "unit size or distance does not match the part number",
}


def image_stats(
    rgb: np.ndarray,
    lighting: str,
    diameter_mm: float,
    cfg: dict[str, Any],
    bounds: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Quality statistics for one face image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    loc = localise_substrate(gray, cfg["classical"], lighting)
    r, _ = loc.polar(gray.shape)
    face = r < 0.9
    vals = gray[face].astype(np.float32)
    # Obstructions and glare are areas; a median filter keeps them but removes
    # isolated pixels from noise or repeated JPEG compression.
    smooth = cv2.medianBlur(gray, cfg["quality"]["area_median_px"])[face].astype(np.float32)
    size = gray.shape[0]
    expected = 0.5 * diameter_mm / cfg["generator"]["fov_mm"] * size
    lo, hi = bounds if bounds else (0.0, 255.0)
    return {
        "brightness": float(np.median(vals)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_32F)[face].var()),
        "dark_area": float((smooth < lo).mean()),
        "bright_area": float((smooth > hi).mean()),
        "offset": float(np.hypot(loc.cx - size / 2, loc.cy - size / 2) / size),
        "radius_ratio": float(loc.radius / expected),
    }


def fit_reference(images: list[tuple[np.ndarray, str, float]], cfg: dict[str, Any]) -> dict:
    """Acceptable ranges per lighting mode from training images ``(rgb, lighting, diameter_mm)``."""
    q = cfg["quality"]
    by_light: dict[str, list[tuple[np.ndarray, float]]] = {}
    for rgb, lighting, diameter in images:
        by_light.setdefault(lighting, []).append((rgb, diameter))
    reference: dict[str, Any] = {"pixel_bounds": {}, "ranges": {}}
    for light, items in by_light.items():
        pixels = np.concatenate(
            [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[::4, ::4].ravel() for rgb, _ in items]
        )
        bounds = tuple(float(np.percentile(pixels, p)) for p in q["pixel_percentiles"])
        stats = [image_stats(rgb, light, d, cfg, bounds) for rgb, d in items]
        ranges = {}
        for s in STATS:
            v = np.array([x[s] for x in stats])
            lo, hi = np.percentile(v, q["image_percentiles"])
            span = max(hi - lo, 1e-6)
            ranges[s] = (float(lo - q["margin"] * span), float(hi + q["margin"] * span))
        reference["pixel_bounds"][light] = bounds
        reference["ranges"][light] = ranges
    return reference


def check_images(
    images: dict[str, np.ndarray],
    lighting: dict[str, str],
    diameter_mm: float,
    reference: dict,
    cfg: dict[str, Any],
) -> list[str]:
    """Human-readable quality problems for both faces (empty if acceptable)."""
    issues = []
    for view, rgb in images.items():
        light = lighting.get(view, "diffuse")
        if light not in reference["ranges"]:
            issues.append(f"{view} image: no quality reference for lighting mode {light!r}")
            continue
        stats = image_stats(rgb, light, diameter_mm, cfg, reference["pixel_bounds"][light])
        for s in STATS:
            lo, hi = reference["ranges"][light][s]
            one_sided_ok = s in ("dark_area", "bright_area", "offset") and stats[s] <= hi
            if not one_sided_ok and not lo <= stats[s] <= hi:
                issues.append(f"{view} image: {LABELS[s]}")
    return issues


def reference_path(run_path: Path) -> Path:
    """Where a run stores its quality reference."""
    return run_path / "models" / "image_quality.json"
