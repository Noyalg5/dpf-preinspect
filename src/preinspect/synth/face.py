"""Base face rendering: fixture, steel can, mat, skin and plugged honeycomb.

The renderer works on physical layers rather than pixels of colour, so each
lighting mode can be derived from the same scene:

- ``albedo``: surface reflectance (linear RGB),
- ``height``: relief used for low-angle shading,
- ``transmit``: light passing through the substrate under backlight
  (only through-channels, cracks and chips transmit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

VIEW_PARITY = {"inlet": 0, "outlet": 1}

# Relative surface heights used by the low-angle shading model.
HEIGHT = {
    "fixture": 0.3,
    "can": 0.8,
    "mat": 0.0,
    "skin": 1.0,
    "wall": 1.0,
    "plug": 0.9,
    "open": 0.0,
}


@dataclass
class Pose:
    """Placement of the substrate in the (render-resolution) image."""

    cx: float
    cy: float
    radius_px: float
    rotation: float

    def scaled(self, factor: float) -> Pose:
        """Return the pose at a different image resolution."""
        return Pose(self.cx * factor, self.cy * factor, self.radius_px * factor, self.rotation)


@dataclass
class Layers:
    """Per-pixel physical layers plus ground-truth masks for one view."""

    albedo: np.ndarray
    height: np.ndarray
    transmit: np.ndarray
    gloss: np.ndarray
    u: np.ndarray
    v: np.ndarray
    r: np.ndarray
    cell_i: np.ndarray
    cell_j: np.ndarray
    wall: np.ndarray
    plugged: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def face(self) -> np.ndarray:
        """Pixels on the substrate (inside the outer skin edge)."""
        return self.r < 1.0

    def add_mask(self, cls: str, mask: np.ndarray) -> None:
        """Union a defect mask into the per-class ground truth."""
        self.masks[cls] = self.masks.get(cls, np.zeros_like(mask, dtype=bool)) | mask


def frame_coords(size: int, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """Map every pixel centre to substrate coordinates ``(u, v)``."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    dx, dy = xs - pose.cx, ys - pose.cy
    c, s = math.cos(pose.rotation), math.sin(pose.rotation)
    return (c * dx + s * dy) / pose.radius_px, (-s * dx + c * dy) / pose.radius_px


def to_pixels(points: list[list[float]] | np.ndarray, pose: Pose) -> np.ndarray:
    """Map substrate coordinates to pixel coordinates (inverse of :func:`frame_coords`)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    c, s = math.cos(pose.rotation), math.sin(pose.rotation)
    x = pose.cx + pose.radius_px * (c * pts[:, 0] - s * pts[:, 1])
    y = pose.cy + pose.radius_px * (s * pts[:, 0] + c * pts[:, 1])
    return np.stack([x, y], axis=1)


def lowfreq_noise(rng: np.random.Generator, size: int, grid: int = 8) -> np.ndarray:
    """Smooth random field in [0, 1] (bicubic upsampling of a coarse grid)."""
    coarse = rng.random((grid, grid)).astype(np.float32)
    return np.clip(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def base_layers(
    scene: dict[str, Any],
    view: str,
    u: np.ndarray,
    v: np.ndarray,
    gcfg: dict[str, Any],
    rng: np.random.Generator,
    wall_boost: np.ndarray | None = None,
) -> Layers:
    """Render the defect-free face for one view.

    Parameters
    ----------
    scene : dict
        Scene description from :func:`preinspect.synth.scene.sample_scene`.
    view : str
        ``"inlet"`` or ``"outlet"``.
    u, v : numpy.ndarray
        Substrate coordinates per pixel (possibly warped by melt damage).
    gcfg : dict
        Generator configuration.
    rng : numpy.random.Generator
        Texture generator.
    wall_boost : numpy.ndarray, optional
        Extra wall thickness per pixel (fused cells in melted regions).

    Returns
    -------
    Layers
        Physical layers with empty defect masks.
    """
    app = gcfg["appearance"]
    col = {k: np.asarray(c, dtype=np.float32) for k, c in app["colours"].items()}
    size = u.shape[0]
    r = np.hypot(u, v)
    mat = scene["material"]

    pitch = 2.0 / scene["n_cells"]
    cu, cv = (u + 1.0) / pitch, (v + 1.0) / pitch
    ci, cj = np.floor(cu), np.floor(cv)
    fu, fv = cu - ci, cv - cj
    half = app["wall_fraction"] / 2.0 + (0.0 if wall_boost is None else wall_boost)
    wall = (fu < half) | (fu > 1 - half) | (fv < half) | (fv > 1 - half)
    plugged = (np.mod(ci + cj + VIEW_PARITY[view], 2) == 0) & ~wall
    open_ = ~wall & ~plugged

    albedo = np.empty((size, size, 3), dtype=np.float32)
    height = np.empty((size, size), dtype=np.float32)
    tex = lowfreq_noise(rng, size)
    skin_w = app["skin_width"]
    in_face = r < 1.0 - skin_w
    regions = [
        (r >= app["can_outer"], col["fixture"], HEIGHT["fixture"]),
        ((r >= 1.0 + app["mat_gap"]) & (r < app["can_outer"]), col["can"], HEIGHT["can"]),
        ((r >= 1.0) & (r < 1.0 + app["mat_gap"]), col["mat"], HEIGHT["mat"]),
        ((r >= 1.0 - skin_w) & (r < 1.0), col[f"skin_{mat}"], HEIGHT["skin"]),
        (in_face & wall, col[f"{mat}_wall"], HEIGHT["wall"]),
        (in_face & plugged, col[f"{mat}_plug"], HEIGHT["plug"]),
        (in_face & open_, col["channel"], HEIGHT["open"]),
    ]
    for region, colour, h in regions:
        albedo[region] = colour
        height[region] = h
    if scene["segment_cells"]:
        seg, jf = scene["segment_cells"], app["sic_joint_fraction"]
        joint = in_face & (
            ((np.mod(ci, seg) == 0) & (fu < jf)) | ((np.mod(cj, seg) == 0) & (fv < jf))
        )
        albedo[joint] = col["sic_joint"]
        height[joint] = HEIGHT["wall"]

    can = (r >= 1.0 + app["mat_gap"]) & (r < app["can_outer"])
    streaks = np.interp(r, np.linspace(0, 2, 96), rng.uniform(0.85, 1.15, 96)).astype(np.float32)
    albedo[can] *= streaks[can, None]
    albedo *= (0.92 + 0.16 * tex)[..., None]

    film = scene["soot_film"][view] * (0.6 + 0.8 * lowfreq_noise(rng, size, grid=6))
    film = np.where(r < 1.0, film, 0.0)[..., None]
    albedo = albedo * (1.0 - film) + col["soot"] * film

    zeros = np.zeros((size, size), dtype=np.float32)
    return Layers(
        albedo=albedo,
        height=height,
        transmit=zeros.copy(),
        gloss=zeros.copy(),
        u=u,
        v=v,
        r=r,
        cell_i=ci.astype(np.int32),
        cell_j=cj.astype(np.int32),
        wall=wall,
        plugged=plugged & in_face,
    )
