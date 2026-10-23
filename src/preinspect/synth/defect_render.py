"""Defect and nuisance rasterisation onto the physical layers of a face.

Each function updates ``Layers`` in place and, for defects, records a
ground-truth mask. Nuisances (dust, marker ink) deliberately record no mask:
they must not be reported as defects.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from preinspect.synth.face import Layers, Pose, lowfreq_noise, to_pixels

CELL_KEY = 100_000


def _colour(gcfg: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(gcfg["appearance"]["colours"][name], dtype=np.float32)


def blob(
    u: np.ndarray, v: np.ndarray, centre: list[float], radius: float, harm: list[list[float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Irregular blob around ``centre``.

    Returns
    -------
    inside : numpy.ndarray of bool
        Pixels inside the perturbed boundary.
    falloff : numpy.ndarray of float
        1 in the core, falling to 0 at the boundary.
    """
    du, dv = u - centre[0], v - centre[1]
    rho = np.hypot(du, dv)
    ang = np.arctan2(dv, du)
    edge = np.full_like(rho, radius)
    for k, (amp, phase) in enumerate(harm, start=2):
        edge += radius * amp * np.cos(k * ang + phase)
    falloff = np.clip((edge - rho) / (0.35 * edge), 0.0, 1.0)
    return rho < edge, falloff


def melt_warp(
    u: np.ndarray, v: np.ndarray, defect: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distort cell coordinates inside a melted region (applied before the honeycomb)."""
    cu, cv = defect["centre"]
    du, dv = u - cu, v - cv
    rho = np.hypot(du, dv)
    weight = np.clip(1.0 - rho / defect["radius"], 0.0, 1.0) ** 2
    swirl = defect["strength"] * 2.5 * weight
    c, s = np.cos(swirl), np.sin(swirl)
    u2 = cu + c * du - s * dv
    v2 = cv + s * du + c * dv
    return u2.astype(np.float32), v2.astype(np.float32), (0.18 * weight).astype(np.float32)


def apply_melt(L: Layers, d: dict[str, Any], gcfg: dict[str, Any]) -> None:
    """Glazed, fused appearance over a melted region."""
    _, f = blob(L.u, L.v, d["centre"], d["radius"], [])
    a = (0.7 * f)[..., None]
    L.albedo = L.albedo * (1 - a) + _colour(gcfg, "melt") * a
    L.gloss = np.maximum(L.gloss, f)
    L.add_mask("melt", (f > 0.15) & L.face)


def apply_missing_plugs(L: Layers, d: dict[str, Any], gcfg: dict[str, Any]) -> None:
    """Plugged cells that have lost their plug render as open, through channels."""
    targets = np.array([i * CELL_KEY + j for i, j in d["cells"]], dtype=np.int64)
    ids = L.cell_i.astype(np.int64) * CELL_KEY + L.cell_j
    hit = np.isin(ids, targets) & L.plugged
    L.albedo[hit] = _colour(gcfg, "channel")
    L.height[hit] = 0.0
    L.transmit[hit] = 1.0
    L.add_mask("missing_plug", hit)


def apply_face_plugging(L: Layers, d: dict[str, Any], gcfg: dict[str, Any], rng) -> None:
    """Deposits blocking channel mouths: soot cake, ash fill or coke crust."""
    inside, f = blob(L.u, L.v, d["centre"], d["radius"], d["harm"])
    region = inside & (L.r < 1.0)
    tex = lowfreq_noise(rng, L.r.shape[0], grid=24)
    if d["deposit"] == "ash":
        fill = region & ~L.plugged & ~L.wall
        L.albedo[fill] = _colour(gcfg, "ash") * (0.85 + 0.3 * tex[fill, None])
        a = 0.35 * region[..., None]
        L.albedo = L.albedo * (1 - a) + _colour(gcfg, "ash") * a
        L.height[fill] = 0.85
    else:
        colour = _colour(gcfg, "soot" if d["deposit"] == "soot" else "coke")
        a = (np.clip(0.75 + 0.5 * f, 0, 0.95) * region)[..., None]
        L.albedo = L.albedo * (1 - a) + colour * (0.7 + 0.6 * tex[..., None]) * a
        L.height[region] = 0.9 + 0.3 * (tex[region] if d["deposit"] == "coke" else 0.0)
    L.add_mask("face_plugging", region)


def apply_stain(L: Layers, d: dict[str, Any], gcfg: dict[str, Any]) -> None:
    """Oil darkens and yellows; coolant leaves a pale, crusty deposit."""
    _, f = blob(L.u, L.v, d["centre"], d["radius"], d["harm"])
    f = f * (L.r < 1.0)
    a = (d["opacity"] * f)[..., None]
    fluid = _colour(gcfg, d["fluid"])
    if d["fluid"] == "oil":
        L.albedo = L.albedo * (1 - a * (1 - fluid))
    else:
        L.albedo = L.albedo * (1 - 0.8 * a) + fluid * 0.8 * a
    L.add_mask("stain", f > 0.5)


def apply_outlet_soot(L: Layers, d: dict[str, Any], gcfg: dict[str, Any], rng) -> None:
    """Soot carried through leak paths and deposited on the outlet face."""
    inside, f = blob(L.u, L.v, d["centre"], d["radius"], d["harm"])
    streak = lowfreq_noise(rng, L.r.shape[0], grid=16)
    density = d["density"] * f * (0.6 + 0.4 * streak) * (L.r < 1.0)
    a = density[..., None]
    L.albedo = L.albedo * (1 - a) + _colour(gcfg, "soot") * a
    L.add_mask("outlet_soot", (density > 0.3) & inside)


def apply_crack(L: Layers, d: dict[str, Any], pose: Pose) -> None:
    """A crack is a narrow gap: dark under front light, bright under backlight."""
    size = L.r.shape[0]
    canvas = np.zeros((size, size), dtype=np.uint8)
    pts = np.round(to_pixels(d["points"], pose)).astype(np.int32)
    thickness = max(1, int(round(d["width"] * pose.radius_px)))
    cv2.polylines(canvas, [pts], isClosed=False, color=1, thickness=thickness)
    gap = (canvas > 0) & (L.r < 1.0)
    halo = cv2.dilate(canvas, np.ones((3, 3), np.uint8), iterations=2) > 0
    L.albedo[halo & (L.r < 1.0)] *= 0.75
    L.albedo[gap] *= 0.12
    L.height[gap] = -1.0
    L.transmit[gap] = 0.8
    L.add_mask("crack", gap)


def apply_chip(L: Layers, d: dict[str, Any], gcfg: dict[str, Any]) -> None:
    """Missing material at the periphery, exposing the mat behind."""
    ang = np.arctan2(L.v, L.u)
    delta = np.abs(np.angle(np.exp(1j * (ang - d["angle"]))))
    # float32 rounding can push cos(pi/2) slightly below zero.
    shape = np.clip(np.cos(0.5 * math.pi * np.clip(delta / d["half_width"], 0, 1)), 0, 1) ** 0.7
    for k, (amp, phase) in enumerate(d["harm"], start=3):
        shape *= 1.0 + amp * np.cos(k * ang + phase)
    chipped = (L.r < 1.0) & (L.r > 1.0 - d["depth"] * shape) & (delta < d["half_width"])
    L.albedo[chipped] = _colour(gcfg, "mat") * 0.6
    L.height[chipped] = -1.0
    L.transmit[chipped] = 0.5
    L.add_mask("chip", chipped)


def apply_debris(L: Layers, d: dict[str, Any], gcfg: dict[str, Any], pose: Pose) -> None:
    """Foreign objects lying on the face."""
    size = L.r.shape[0]
    canvas = np.zeros((size, size), dtype=np.uint8)
    for item in d["items"]:
        rng = np.random.default_rng(item["seed"])
        angles = np.sort(rng.uniform(0, 2 * math.pi, 7))
        radii = item["size"] * rng.uniform(0.5, 1.2, 7)
        cu, cv = item["centre"]
        poly = [
            [cu + r * math.cos(a), cv + r * math.sin(a)] for r, a in zip(radii, angles, strict=True)
        ]
        item_mask = np.zeros_like(canvas)
        cv2.fillPoly(item_mask, [np.round(to_pixels(poly, pose)).astype(np.int32)], 1)
        sel = item_mask > 0
        L.albedo[sel] = _colour(gcfg, f"debris_{item['kind']}") * rng.uniform(0.8, 1.1)
        canvas |= item_mask
    sel = canvas > 0
    L.height[sel] = 1.5
    L.transmit[sel] = 0.0
    L.add_mask("debris", sel)


def apply_dust(L: Layers, n: dict[str, Any], gcfg: dict[str, Any], pose: Pose) -> None:
    """Fine light speckles: a nuisance, never a defect."""
    rng = np.random.default_rng(n["seed"])
    size = L.r.shape[0]
    canvas = np.zeros((size, size), dtype=np.uint8)
    rho = 1.08 * np.sqrt(rng.random(n["count"]))
    theta = rng.uniform(0, 2 * math.pi, n["count"])
    pts = to_pixels(np.stack([rho * np.cos(theta), rho * np.sin(theta)], 1), pose)
    for (x, y), rad in zip(pts, rng.integers(1, 3, n["count"]), strict=True):
        cv2.circle(canvas, (int(x), int(y)), int(rad), 1, -1)
    a = 0.5 * (canvas > 0)[..., None]
    L.albedo = L.albedo * (1 - a) + _colour(gcfg, "dust") * a


def apply_marker(L: Layers, n: dict[str, Any], gcfg: dict[str, Any], pose: Pose) -> None:
    """Hand-written marker ink on the can or skin: a nuisance."""
    rng = np.random.default_rng(n["seed"])
    size = L.r.shape[0]
    canvas = np.zeros((size, size), dtype=np.uint8)
    tangent = n["angle"] + math.pi / 2
    for k in range(int(rng.integers(3, 6))):
        a0 = n["angle"] + 0.06 * k
        base = np.array([math.cos(a0), math.sin(a0)]) * n["radius"]
        steps = (
            rng.normal(0, 0.025, (4, 2)) + np.array([math.cos(tangent), math.sin(tangent)]) * 0.012
        )
        pts = np.round(to_pixels(base + np.cumsum(steps, axis=0), pose)).astype(np.int32)
        cv2.polylines(canvas, [pts], False, 1, thickness=max(1, int(0.012 * pose.radius_px)))
    L.albedo[canvas > 0] = _colour(gcfg, n["ink"])
