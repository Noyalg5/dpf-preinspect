"""Accumulators for per-class segmentation and anomaly-detection metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _auc(labels: list, scores: list) -> float | None:
    y = np.asarray(labels, dtype=bool)
    if y.all() or not y.any():
        return None
    return float(roc_auc_score(y, np.asarray(scores)))


class SegmentationAccumulator:
    """Dice, IoU and image-level AUROC per class over many images.

    Dice and IoU are computed on pixels pooled over all images (so images
    without the class contribute their false positives); image-level AUROC
    uses the presence head.
    """

    def __init__(self, classes: list[str]) -> None:
        self.classes = classes
        n = len(classes)
        self.inter, self.pred, self.true = np.zeros(n), np.zeros(n), np.zeros(n)
        self.presence: list[list[float]] = [[] for _ in classes]
        self.labels: list[list[bool]] = [[] for _ in classes]

    def update(self, prob: np.ndarray, truth: np.ndarray, presence: np.ndarray, thr: float) -> None:
        """Add one image: ``prob`` and ``truth`` are ``(C, H, W)``."""
        pred = prob > thr
        flat_p, flat_t = pred.reshape(len(self.classes), -1), truth.reshape(len(self.classes), -1)
        self.inter += (flat_p & flat_t).sum(axis=1)
        self.pred += flat_p.sum(axis=1)
        self.true += flat_t.sum(axis=1)
        for k in range(len(self.classes)):
            self.presence[k].append(float(presence[k]))
            self.labels[k].append(bool(flat_t[k].any()))

    def summary(self) -> dict[str, dict[str, float | None]]:
        """Per-class Dice, IoU, image AUROC and number of positive images."""
        out = {}
        for k, cls in enumerate(self.classes):
            denom = self.pred[k] + self.true[k]
            union = denom - self.inter[k]
            out[cls] = {
                "dice": float(2 * self.inter[k] / denom) if denom else None,
                "iou": float(self.inter[k] / union) if union else None,
                "image_auroc": _auc(self.labels[k], self.presence[k]),
                "positive_images": int(np.sum(self.labels[k])),
            }
        return out


class AnomalyAccumulator:
    """Image- and pixel-level AUROC of anomaly maps against defect masks.

    Pixels are subsampled on a regular grid to keep memory small.
    """

    def __init__(self, stride: int = 4) -> None:
        self.stride = stride
        self.img_scores: list[float] = []
        self.img_labels: list[bool] = []
        self.pix_scores: list[np.ndarray] = []
        self.pix_labels: list[np.ndarray] = []

    def update(self, amap: np.ndarray, defect: np.ndarray) -> None:
        """Add one image's anomaly map and its any-defect mask."""
        self.img_scores.append(float(amap.max()))
        self.img_labels.append(bool(defect.any()))
        s = self.stride
        self.pix_scores.append(amap[::s, ::s].ravel())
        self.pix_labels.append(defect[::s, ::s].ravel())

    def summary(self) -> dict[str, float | None]:
        """Image AUROC, pixel AUROC and counts."""
        pix_auc = None
        if self.pix_scores:
            pix_auc = _auc(
                np.concatenate(self.pix_labels).tolist(), np.concatenate(self.pix_scores).tolist()
            )
        return {
            "image_auroc": _auc(self.img_labels, self.img_scores),
            "pixel_auroc": pix_auc,
            "images": len(self.img_scores),
            "defective_images": int(np.sum(self.img_labels)),
        }
