"""Render one face view of a unit: scene + acquisition -> image and masks.

Rendering happens at ``supersample`` times the output size and is
area-averaged down, which avoids aliasing of the fine cell grid.
Masks are stored as an 8-bit bitmask: bit ``k`` is defect class ``k``.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from preinspect.synth import defect_render as dr
from preinspect.synth.acquisition import camera_response, clamp_mask, make_pose, shade
from preinspect.synth.face import base_layers, frame_coords
from preinspect.utils.seeding import derive_seed


def render_view(
    gcfg: dict[str, Any],
    classes: list[str],
    scene: dict[str, Any],
    view: str,
    acq: dict[str, Any],
    out_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Render one view.

    Parameters
    ----------
    gcfg : dict
        Generator configuration.
    classes : list of str
        Defect class names; position defines the mask bit.
    scene : dict
        Scene description.
    view : str
        ``"inlet"`` or ``"outlet"``.
    acq : dict
        Acquisition record.
    out_size : int
        Output image side in pixels.

    Returns
    -------
    image : numpy.ndarray
        ``(out_size, out_size, 3)`` uint8 RGB.
    mask : numpy.ndarray
        ``(out_size, out_size)`` uint8 bitmask of defect classes.
    info : dict
        Pose at output resolution and the occluded share of the face.
    """
    ss = int(gcfg["supersample"])
    size = out_size * ss
    pose = make_pose(acq, size, scene["diameter_mm"], gcfg["fov_mm"])
    u, v = frame_coords(size, pose)
    defects = [d for d in scene["defects"] if d["view"] == view]
    wall_boost = None
    for d in defects:
        if d["cls"] == "melt":
            u, v, wall_boost = dr.melt_warp(u, v, d)
    rng = np.random.default_rng(derive_seed(scene["texture_seed"], view))
    L = base_layers(scene, view, u, v, gcfg, rng, wall_boost)

    by_cls: dict[str, list[dict[str, Any]]] = {}
    for d in defects:
        by_cls.setdefault(d["cls"], []).append(d)
    for d in by_cls.get("melt", []):
        dr.apply_melt(L, d, gcfg)
    for d in by_cls.get("missing_plug", []):
        dr.apply_missing_plugs(L, d, gcfg)
    for d in by_cls.get("face_plugging", []):
        dr.apply_face_plugging(L, d, gcfg, rng)
    for d in by_cls.get("stain", []):
        dr.apply_stain(L, d, gcfg)
    for d in by_cls.get("outlet_soot", []):
        dr.apply_outlet_soot(L, d, gcfg, rng)
    for d in by_cls.get("crack", []):
        dr.apply_crack(L, d, pose)
    for d in by_cls.get("chip", []):
        dr.apply_chip(L, d, gcfg)
    for d in by_cls.get("debris", []):
        dr.apply_debris(L, d, gcfg, pose)
    for n in scene["nuisances"]:
        if n["view"] == view:
            fn = dr.apply_dust if n["kind"] == "dust" else dr.apply_marker
            fn(L, n, gcfg, pose)

    radiance = shade(L, acq, gcfg, pose)
    occluded = clamp_mask(L, acq)
    image = camera_response(radiance, occluded, acq, gcfg, pose, out_size)

    occ_small = cv2.resize(occluded.astype(np.float32), (out_size, out_size), cv2.INTER_AREA) > 0.5
    thr = gcfg["mask_area_threshold"]
    kernel = np.ones((3, 3), np.uint8)
    bitmask = np.zeros((out_size, out_size), dtype=np.uint8)
    for k, cls in enumerate(classes):
        if cls not in L.masks:
            continue
        m = cv2.resize(L.masks[cls].astype(np.float32), (out_size, out_size), cv2.INTER_AREA) > thr
        if cls == "crack" and gcfg["crack_mask_dilate_px"] > 0:
            m = cv2.dilate(m.astype(np.uint8), kernel, iterations=gcfg["crack_mask_dilate_px"]) > 0
        bitmask |= ((m & ~occ_small).astype(np.uint8) << k).astype(np.uint8)

    small = pose.scaled(1.0 / ss)
    info = {
        "cx": small.cx,
        "cy": small.cy,
        "radius_px": small.radius_px,
        "rotation_deg": math.degrees(small.rotation),
        "mm_per_px": scene["diameter_mm"] / (2.0 * small.radius_px),
        "occluded_face_share": float((occluded & L.face).sum() / max(1, L.face.sum())),
    }
    return image, bitmask, info
