"""PNG overlays of predicted defect masks and anomaly heatmaps for the UI."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from preinspect.report.style import SERIES


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))


def to_png_base64(image: np.ndarray) -> str:
    """Encode an RGB uint8 image as base64 PNG."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("ascii")


def defect_overlay(image: np.ndarray, prob: np.ndarray, threshold: float) -> np.ndarray:
    """Outline each predicted defect class in its palette colour."""
    out = image.copy()
    for k in range(prob.shape[0]):
        mask = (prob[k] > threshold).astype(np.uint8)
        if mask.any():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, contours, -1, _rgb(SERIES[k % len(SERIES)]), 1)
    return out


def anomaly_overlay(image: np.ndarray, amap: np.ndarray, threshold: float) -> np.ndarray:
    """Blend a heatmap of the anomaly map, saturating at the guard threshold."""
    scaled = np.clip(amap / max(threshold, 1e-6), 0, 1)
    heat = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    alpha = (0.6 * scaled)[..., None]
    return (image * (1 - alpha) + heat * alpha).astype(np.uint8)


def overlays(
    images: dict[str, np.ndarray], maps: dict[str, dict], threshold: float, anomaly_threshold: float
) -> dict[str, dict[str, str]]:
    """Base64 PNG overlays per view: ``defects`` and ``anomaly``."""
    return {
        view: {
            "defects": to_png_base64(defect_overlay(images[view], m["prob"], threshold)),
            "anomaly": to_png_base64(
                anomaly_overlay(images[view], m["anomaly"], anomaly_threshold)
            ),
        }
        for view, m in maps.items()
    }
