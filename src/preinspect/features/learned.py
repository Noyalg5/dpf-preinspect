"""Concepts measured from predicted defect masks and anomaly maps.

The same measurements an inspector would quote (crack length in mm and its
clock position, number of missing plugs, stained area) are read off the
U-Net's probability maps, so the outcome model stays interpretable even
though the masks come from a deep network.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from preinspect.classical.localise import Localisation


def clock_hour(x: float, y: float, cx: float, cy: float) -> float:
    """Clock position (12 at the top of the image, clockwise) of a point."""
    angle = math.degrees(math.atan2(x - cx, -(y - cy))) % 360.0
    hour = angle / 30.0
    return 12.0 if hour < 0.5 else hour


def learned_concepts(
    prob: np.ndarray,
    presence: np.ndarray,
    classes: list[str],
    loc: Localisation,
    mm_per_px: float,
    threshold: float,
) -> dict[str, float]:
    """Concept measurements for one image.

    Parameters
    ----------
    prob : numpy.ndarray
        ``(C, H, W)`` class probabilities.
    presence : numpy.ndarray
        ``(C,)`` image-level presence probabilities.
    classes : list of str
        Class names in channel order.
    loc : Localisation
        Substrate circle (for areas relative to the face and clock positions).
    mm_per_px : float
        Scale from the known substrate diameter.
    threshold : float
        Mask binarisation threshold.
    """
    face_area = math.pi * loc.radius**2
    out: dict[str, float] = {}
    for k, cls in enumerate(classes):
        mask = prob[k] > threshold
        out[f"{cls}_area"] = float(mask.sum() / face_area)
        out[f"{cls}_peak"] = float(prob[k].max())
        out[f"{cls}_presence"] = float(presence[k])
        if cls in ("crack", "chip"):
            regions = regionprops(label(mask))
            if regions:
                big = max(regions, key=lambda r: r.area)
                y, x = big.centroid
                out[f"{cls}_clock"] = clock_hour(x, y, loc.cx, loc.cy)
                out[f"{cls}_radius"] = math.hypot(x - loc.cx, y - loc.cy) / loc.radius
            else:
                out[f"{cls}_clock"] = float("nan")
                out[f"{cls}_radius"] = float("nan")
        if cls == "crack":
            out["crack_length_mm"] = float(skeletonize(mask).sum() * mm_per_px)
        if cls in ("missing_plug", "debris"):
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            out[f"{cls}_count"] = float((stats[1:, cv2.CC_STAT_AREA] >= 2).sum()) if n > 1 else 0.0
    return out


def anomaly_concepts(
    amap: np.ndarray, prob: np.ndarray, threshold: float, dilate_px: int
) -> dict[str, float]:
    """Overall anomaly and the part not explained by any known defect class.

    The unexplained score masks out (dilated) regions where the perception
    model already names a defect; what remains anomalous is something the
    model has no class for, such as a novel defect or an unusual set-up.
    """
    known = (prob > threshold).any(axis=0).astype(np.uint8)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        known = cv2.dilate(known, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return {
        "max": float(amap.max()),
        "unexplained": float((amap * (1 - known)).max()),
    }
