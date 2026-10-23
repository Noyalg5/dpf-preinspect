"""Measured "concepts" from one face image by classical image processing.

Every concept is a physical measurement an inspector would recognise:
crack length in millimetres, missing-plug count, blocked-cell fraction,
perimeter deviation (chips), stained area, dark area on the face (outlet
soot when measured on the outlet view), plus image-quality indicators.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from skimage.filters import sato
from skimage.measure import label
from skimage.morphology import skeletonize

from preinspect.classical.cells import cell_summary, classify_cells, sample_cells
from preinspect.classical.lattice import estimate_lattice, expected_pitch_px, remove_lattice
from preinspect.classical.localise import Localisation, localise_substrate


def crack_concepts(
    residual: np.ndarray,
    evaluate: np.ndarray,
    pitch: float,
    lighting: str,
    mm_per_px: float,
    ccfg: dict[str, Any],
) -> tuple[dict[str, float], np.ndarray]:
    """Ridge filter on the de-gridded face, then skeleton length of long ridges.

    Ridges are evaluated inside an eroded face so the boundary of the
    filtered region does not register as a crack.

    Returns
    -------
    concepts : dict
        Crack length (mm), segment count, longest segment (mm), ridge peak.
    crack_mask : numpy.ndarray of bool
        Detected crack pixels, dilated by about one cell.
    """
    cfg = ccfg["crack"]
    ridge = sato(residual, sigmas=cfg["sato_sigmas"], black_ridges=lighting != "backlight")
    vals = ridge[evaluate]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    binary = (ridge > med + cfg["k_mad"] * mad) & evaluate
    labels = label(skeletonize(binary), connectivity=2)
    lengths = np.bincount(labels.ravel())
    lengths[0] = 0
    keep = lengths >= cfg["min_length_cells"] * pitch
    crack = keep[labels]
    long = lengths[keep]
    k = max(3, int(round(pitch)) | 1)
    crack_mask = cv2.dilate(crack.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return {
        "crack_length_mm": float(long.sum() * mm_per_px),
        "crack_segments": float(long.size),
        "crack_max_segment_mm": float(long.max() * mm_per_px) if long.size else 0.0,
        "ridge_peak": float((np.percentile(vals, 99.5) - med) / mad),
    }, crack_mask


def chip_concepts(
    gray: np.ndarray, loc: Localisation, lighting: str, ccfg: dict[str, Any]
) -> dict[str, float]:
    """Perimeter deviation: angular sectors where the outer ring of the face is missing."""
    cfg = ccfg["chip"]
    n_ang = cfg["n_angles"]
    max_r = cfg["r_stop"] * loc.radius
    n_rad = int(max_r)
    polar = cv2.warpPolar(
        gray.astype(np.float32), (n_rad, n_ang), (loc.cx, loc.cy), max_r, cv2.WARP_POLAR_LINEAR
    )
    # Smooth along the angle only lightly and barely along the radius: chips
    # are a few pixels deep and vanish under cell-scale radial smoothing.
    radial = cfg["radial_sigma_px"] * gray.shape[0] / ccfg["ref_size"]
    polar = cv2.GaussianBlur(polar, (0, 0), sigmaX=radial, sigmaY=1.0)
    r_norm = np.arange(n_rad) / n_rad * cfg["r_stop"]
    ring = (r_norm >= cfg["r_start"]) & (r_norm < cfg["ring_stop"])
    can = (r_norm >= cfg["can_range"][0]) & (r_norm < cfg["can_range"][1])
    ref = np.median(polar, axis=0, keepdims=True) + 1e-3
    dev = (polar - ref) / ref
    flagged = dev > cfg["bright_dev"] if lighting == "backlight" else dev < cfg["dark_dev"]
    # A clamp jaw darkens the can as well as the face edge; a chip does not.
    occluded = (dev[:, can] < cfg["dark_dev"]).mean(axis=1) > cfg["occluded_share"]
    depth = flagged[:, ring].mean(axis=1) * (cfg["ring_stop"] - cfg["r_start"])
    depth[occluded] = 0.0
    chipped = depth > cfg["depth_threshold"]
    return {
        "chip_depth_max": float(depth.max()),
        "chip_angle_fraction": float(chipped.mean()),
        "clamp_angle_fraction": float(occluded.mean()),
    }


def colour_concepts(
    rgb: np.ndarray, face: np.ndarray, pitch: float, ccfg: dict[str, Any]
) -> dict[str, float]:
    """LAB statistics at cell scale: stained (off-hue), dark and bright areas."""
    cfg = ccfg["colour"]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), cfg["smooth_cells"] * pitch)
    lightness, a, b = lab[..., 0][face], lab[..., 1][face], lab[..., 2][face]
    chroma = np.hypot(a - np.median(a), b - np.median(b))
    l_med = float(np.median(lightness)) + 1e-3
    return {
        "dark_p02": float(np.percentile(lightness, 2) / l_med),
        "stain_fraction": float((chroma > cfg["chroma_threshold"]).mean()),
        "chroma_p99": float(np.percentile(chroma, 99)),
        "yellowness_p99": float(np.percentile(b - np.median(b), 99)),
        "dark_fraction": float((lightness < cfg["dark_level"] * l_med).mean()),
        "bright_fraction": float((lightness > cfg["bright_level"] * l_med).mean()),
        "face_brightness": l_med / 255.0,
    }


def image_concepts(
    rgb: np.ndarray, lighting: str, cpsi: float, diameter_mm: float, ccfg: dict[str, Any]
) -> dict[str, float]:
    """All classical concepts for one face image.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB uint8 image.
    lighting : str
        Station lighting mode for this image.
    cpsi, diameter_mm : float
        Part metadata (datasheet cell density and diameter).
    ccfg : dict
        ``classical`` configuration section.

    Returns
    -------
    dict
        Concept name to value; NaN where a concept cannot be measured.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    loc = localise_substrate(gray, ccfg, lighting)
    lat = estimate_lattice(gray, loc, ccfg, expected_pitch_px(ccfg, cpsi, diameter_mm, loc.radius))
    r_norm, _ = loc.polar(gray.shape)
    face = r_norm < ccfg["crack"]["face_r"]
    mm_per_px = diameter_mm / (2.0 * loc.radius)
    residual = remove_lattice(gray.astype(np.float32) / 255.0, face, ccfg)
    cracks, crack_mask = crack_concepts(
        residual, r_norm < ccfg["crack"]["eval_r"], lat.pitch, lighting, mm_per_px, ccfg
    )
    cells = classify_cells(sample_cells(gray, loc, lat, ccfg), lighting, ccfg, crack_mask)
    out = {
        "localised": float(loc.ok),
        "radius_px": loc.radius,
        "pitch_px": lat.pitch,
        "lattice_strength": lat.strength,
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_32F)[face].var()) / 255.0,
        **cell_summary(cells),
        **cracks,
        **chip_concepts(gray, loc, lighting, ccfg),
        **colour_concepts(rgb, face, lat.pitch, ccfg),
    }
    out["residual_energy"] = float(np.sqrt(np.mean(residual[face] ** 2)))
    return {
        k: (float(v) if v is not None and math.isfinite(v) else float("nan"))
        for k, v in out.items()
    }
