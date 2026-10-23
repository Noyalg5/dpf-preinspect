"""Image corruptions for robustness testing, each at five severities.

Severity values live in ``campaign.robustness.kinds`` and are physical:
gradient strength, exposure gain, blur sigma (px), noise sigma, shift as a
share of the image, number of lens-dirt spots and occluded area share.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def corrupt(image: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    """Apply one corruption to an RGB uint8 image."""
    x = image.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    if kind == "lighting":
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(0, 2 * math.pi)
        ramp = ((xs - w / 2) * math.cos(a) + (ys - h / 2) * math.sin(a)) / (w / 2)
        x = x * np.clip(1 + level * ramp, 0.05, None)[..., None]
    elif kind == "exposure":
        x = x * level
    elif kind == "blur":
        x = cv2.GaussianBlur(x, (0, 0), level)
    elif kind == "noise":
        x = x + rng.normal(0, level, x.shape).astype(np.float32)
    elif kind == "offset":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-90, 90) * level, 1.0)
        m[:, 2] += rng.choice([-1, 1], 2) * level * w
        x = cv2.warpAffine(x, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif kind == "lens_dirt":
        dirt = np.zeros((h, w), np.float32)
        for _ in range(int(level)):
            centre = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.circle(dirt, centre, int(rng.uniform(0.03, 0.07) * w), 1.0, -1)
        dirt = cv2.GaussianBlur(dirt, (0, 0), 0.02 * w)
        x = x * (1 - 0.6 * np.clip(dirt, 0, 1))[..., None]
    elif kind == "occlusion":
        side = int(math.sqrt(level) * 0.8 * w)
        x0, y0 = rng.integers(int(0.1 * w), int(0.9 * w) - side, 2)
        x[y0 : y0 + side, x0 : x0 + side] = 0.08
    else:
        raise ValueError(f"unknown corruption {kind}")
    return np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)
