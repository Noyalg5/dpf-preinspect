"""PyTorch datasets for face images, with lighting randomisation.

Augmentation imitates what changes between stations and shifts: exposure,
gamma, white balance, a lighting gradient, sensor noise and slight blur,
plus seating changes (flips and quarter turns are exact symmetries of a
round substrate in a square frame). Masks follow geometric changes only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preinspect.data.io import decode_bitmask, read_image, read_mask


@lru_cache(maxsize=8)
def _centred_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return (xs - w / 2) / (w / 2), (ys - h / 2) / (w / 2)


def photometric(img: np.ndarray, rng: np.random.Generator, acfg: dict[str, Any]) -> np.ndarray:
    """Random exposure, gamma, white balance, gradient, blur and noise on a [0, 1] image."""
    x = img * rng.uniform(*acfg["gain"])
    x = np.clip(x, 0, 1) ** rng.uniform(*acfg["gamma"])
    x = x * rng.uniform(1 - acfg["wb"], 1 + acfg["wb"], 3).astype(np.float32)
    gx, gy = _centred_grid(*x.shape[:2])
    angle = rng.uniform(0, 2 * np.pi)
    ramp = gx * np.cos(angle) + gy * np.sin(angle)
    x = x * (1 + rng.uniform(0, acfg["gradient"]) * ramp)[..., None]
    if rng.random() < acfg["blur_prob"]:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.2))
    x = x + rng.normal(0, rng.uniform(0, acfg["noise"]), x.shape).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def geometric(img: np.ndarray, masks: np.ndarray, rng: np.random.Generator):
    """Random flip and quarter turn applied identically to image and masks."""
    if rng.random() < 0.5:
        img, masks = img[:, ::-1], masks[:, :, ::-1]
    k = int(rng.integers(0, 4))
    img = np.rot90(img, k, axes=(0, 1))
    masks = np.rot90(masks, k, axes=(1, 2))
    return np.ascontiguousarray(img), np.ascontiguousarray(masks)


class FaceDataset(Dataset):
    """One item per (unit, view): image, per-class masks, per-class presence.

    Parameters
    ----------
    ddir : Path
        Dataset folder.
    items : list of (str, str)
        ``(unit_id, view)`` pairs.
    class_bits : list of int
        Bit positions (in the stored bitmask) of the classes to predict.
    augment : dict, optional
        Augmentation settings; None disables augmentation.
    crop : int, optional
        Random square crop size for training.
    seed : int
        Base seed for augmentation.
    """

    def __init__(
        self,
        ddir: Path,
        items: list[tuple[str, str]],
        class_bits: list[int],
        augment: dict[str, Any] | None = None,
        crop: int | None = None,
        seed: int = 0,
    ) -> None:
        self.ddir, self.items, self.bits = Path(ddir), items, class_bits
        self.augment, self.crop, self.seed = augment, crop, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uid, view = self.items[idx]
        img = read_image(self.ddir, uid, view).astype(np.float32) / 255.0
        allm = decode_bitmask(read_mask(self.ddir, uid, view), max(self.bits) + 1)
        masks = allm[self.bits].astype(np.float32)
        if self.augment is not None:
            rng = np.random.default_rng((self.seed, self.epoch, idx))
            img = photometric(img, rng, self.augment)
            img, masks = geometric(img, masks, rng)
            if self.crop and self.crop < img.shape[0]:
                y0, x0 = rng.integers(0, img.shape[0] - self.crop + 1, 2)
                img = img[y0 : y0 + self.crop, x0 : x0 + self.crop]
                masks = masks[:, y0 : y0 + self.crop, x0 : x0 + self.crop]
        presence = (masks.reshape(len(self.bits), -1).max(axis=1) > 0).astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(masks)),
            torch.from_numpy(presence),
        )
