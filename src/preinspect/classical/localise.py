"""Substrate localisation from texture.

The honeycomb face is the only strongly textured region in the image (the
fixture and steel can are smooth), so a blurred high-pass magnitude map,
thresholded with Otsu's method, separates face from surroundings. A circle
is then fitted to the convex hull of the largest region, which bridges
chips, clamp jaws and smooth deposits on the face.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class Localisation:
    """Fitted substrate circle in pixel coordinates."""

    cx: float
    cy: float
    radius: float
    ok: bool

    def polar(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Normalised radius and angle of every pixel relative to the fitted circle."""
        ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32) + 0.5
        dx, dy = xs - self.cx, ys - self.cy
        return np.hypot(dx, dy) / self.radius, np.arctan2(dy, dx)


def fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Algebraic least-squares circle fit (Kasa method).

    Parameters
    ----------
    points : numpy.ndarray
        ``(N, 2)`` array of x, y coordinates.

    Returns
    -------
    tuple of float
        Centre x, centre y and radius.
    """
    x, y = points[:, 0].astype(float), points[:, 1].astype(float)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    (c0, c1, c2), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c0 / 2.0, c1 / 2.0
    return float(cx), float(cy), float(np.sqrt(c2 + cx**2 + cy**2))


def _resample_closed(points: np.ndarray, n: int = 180) -> np.ndarray:
    closed = np.vstack([points, points[:1]]).astype(float)
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def localise_substrate(
    gray: np.ndarray, ccfg: dict[str, Any], lighting: str = "diffuse"
) -> Localisation:
    """Locate the substrate face in a grey-level image.

    Parameters
    ----------
    gray : numpy.ndarray
        ``(H, W)`` uint8 or float image.
    ccfg : dict
        ``classical`` configuration section.
    lighting : str
        Station lighting mode; selects the edge landmark used to refine the radius.

    Returns
    -------
    Localisation
        Fitted circle; ``ok`` is False if no plausible face was found, in
        which case the image centre and a nominal radius are returned.
    """
    cfg = ccfg["localise"]
    scale = gray.shape[0] / ccfg["ref_size"]
    g = gray.astype(np.float32) / 255.0
    high_pass = np.abs(g - cv2.GaussianBlur(g, (0, 0), cfg["hp_sigma"] * scale))
    texture = cv2.GaussianBlur(high_pass, (0, 0), cfg["texture_sigma"] * scale)
    tex8 = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(tex8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    h, w = gray.shape
    fallback = Localisation(w / 2.0, h / 2.0, 0.35 * w, False)
    if n < 2:
        return fallback
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < cfg["min_area_frac"] * h * w:
        return fallback
    region = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))[:, 0, :]
    cx, cy, r = fit_circle(_resample_closed(hull))
    r = refine_radius(g, cx, cy, r, cfg, cfg["refine_landmark"].get(lighting, "dark"))
    return Localisation(cx, cy, r * cfg["radius_scale"], True)


def refine_radius(
    g: np.ndarray, cx: float, cy: float, r0: float, cfg: dict[str, Any], landmark: str
) -> float:
    """Snap the radius to the mat gap between skin and can.

    The texture region overshoots the true edge because the skin, mat and
    can edges are themselves textured. Averaged over all angles, the mat gap
    is the darkest thin ring under front or back light, and the brightest
    ring under low-angle light, where its two edges catch the light.
    """
    ys, xs = np.mgrid[0 : g.shape[0], 0 : g.shape[1]].astype(np.float32) + 0.5
    rr = np.hypot(xs - cx, ys - cy)
    lo, hi = cfg["refine_range"]
    sel = (rr >= lo * r0) & (rr <= hi * r0)
    bins = np.floor(rr[sel]).astype(int)
    sums = np.bincount(bins, weights=g[sel])
    counts = np.bincount(bins)
    valid = counts > 0
    radii = np.flatnonzero(valid)
    profile = sums[valid] / counts[valid]
    if radii.size < 3:
        return r0
    pick = np.argmax if landmark == "bright" else np.argmin
    gap_centre = float(radii[int(pick(profile))]) + 0.5
    return gap_centre / cfg["gap_centre_r"]
