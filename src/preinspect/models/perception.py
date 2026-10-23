"""Training and inference for the defect-perception U-Net."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from preinspect.data.io import decode_bitmask, read_mask
from preinspect.data.torch_data import FaceDataset
from preinspect.models.unet import UNet, segmentation_loss
from preinspect.utils.logging import get_logger

log = get_logger(__name__)


def class_bits(cfg: dict[str, Any]) -> list[int]:
    """Bitmask positions of the perception classes."""
    return [cfg["defect_classes"].index(c) for c in cfg["perception"]["classes"]]


def positive_weights(ddir: Path, items: list[tuple[str, str]], bits: list[int], cap: float):
    """Per-class ratio of negative to positive pixels, capped, from up to 300 masks."""
    pos = np.zeros(len(bits))
    total = 0
    for uid, view in items[:300]:
        m = decode_bitmask(read_mask(ddir, uid, view), max(bits) + 1)[bits]
        pos += m.reshape(len(bits), -1).sum(axis=1)
        total += m[0].size
    return torch.tensor(
        np.clip((total - pos) / np.maximum(pos, 1.0), 1.0, cap), dtype=torch.float32
    )


def train_unet(
    cfg: dict[str, Any], ddir: Path, items: list[tuple[str, str]], device: torch.device, seed: int
) -> tuple[UNet, dict[str, Any]]:
    """Train one U-Net on the given (unit, view) items.

    Returns
    -------
    model : UNet
        Trained network in eval mode.
    history : dict
        Mean loss per epoch and whether ImageNet weights were used.
    """
    p = cfg["perception"]
    bits = class_bits(cfg)
    torch.manual_seed(seed)
    ds = FaceDataset(ddir, items, bits, augment=p["augment"], crop=p["crop"], seed=seed)
    loader = DataLoader(
        ds,
        batch_size=p["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = UNet(len(bits), p["pretrained"]).to(device)
    pos_weight = positive_weights(ddir, items, bits, p["pos_weight_max"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    steps = max(1, p["epochs"] * len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=steps)
    losses = []
    for epoch in range(p["epochs"]):
        ds.epoch = epoch
        model.train()
        running = []
        for img, masks, presence in loader:
            img, masks, presence = img.to(device), masks.to(device), presence.to(device)
            seg, pres = model(img)
            loss = segmentation_loss(seg, pres, masks, presence, pos_weight, p["dice_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running.append(float(loss.detach()))
        losses.append(float(np.mean(running)) if running else float("nan"))
        log.info("unet seed %d epoch %d/%d loss %.4f", seed, epoch + 1, p["epochs"], losses[-1])
    return model.eval(), {"loss": losses, "pretrained": model.encoder.pretrained}


@torch.no_grad()
def predict(
    models: list[UNet], images: torch.Tensor, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Average sigmoid outputs of several models for a batch of ``[0, 1]`` images.

    Returns
    -------
    prob : numpy.ndarray
        ``(B, C, H, W)`` per-pixel class probabilities.
    presence : numpy.ndarray
        ``(B, C)`` image-level presence probabilities.
    """
    prob, presence, _ = predict_with_embedding(models, images, device)
    return prob, presence


@torch.no_grad()
def predict_with_embedding(
    models: list, images: torch.Tensor, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """As :func:`predict`, plus the pooled encoder features (None for ONNX models)."""
    x = images.to(device)
    seg_sum, pres_sum, emb_sum = 0.0, 0.0, 0.0
    for model in models:
        seg, pres = model(x)
        seg_sum = seg_sum + torch.sigmoid(seg)
        pres_sum = pres_sum + torch.sigmoid(pres)
        pooled = getattr(model, "pooled", None)
        emb_sum = emb_sum + pooled if pooled is not None and emb_sum is not None else None
    n = len(models)
    emb = None if emb_sum is None else (emb_sum / n).float().cpu().numpy()
    return (seg_sum / n).float().cpu().numpy(), (pres_sum / n).float().cpu().numpy(), emb


def batches(items: list, size: int) -> Iterator[list]:
    """Consecutive chunks of a list."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
