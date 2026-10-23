"""PatchCore-style anomaly detection (Roth et al., 2022), implemented directly.

Mid-level features of a frozen ImageNet ResNet-18 (layers 2 and 3, locally
averaged) describe each image patch. A memory bank holds patches from
defect-free training images, reduced by greedy coreset selection. A test
patch's anomaly score is its distance to the nearest bank entry.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from preinspect.models.backbone import Trunk, normalise


class PatchFeatures(nn.Module):
    """Frozen layer-2 + layer-3 patch descriptors at stride 8."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.trunk = Trunk(pretrained).eval()
        for param in self.trunk.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 384, H/8, W/8)`` descriptors for ``[0, 1]`` images."""
        _, _, s8, s16, _ = self.trunk(normalise(x))
        s16 = F.interpolate(s16, size=s8.shape[-2:], mode="bilinear", align_corners=False)
        return F.avg_pool2d(torch.cat([s8, s16], dim=1), 3, stride=1, padding=1)


def greedy_coreset(feats: torch.Tensor, k: int, proj_dim: int, seed: int) -> torch.Tensor:
    """Indices of ``k`` rows chosen by greedy k-centre selection on a random projection.

    Runs on the CPU: the loop is inherently sequential, and on an accelerator
    every iteration would wait for a device synchronisation.
    """
    if feats.shape[0] <= k:
        return torch.arange(feats.shape[0])
    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(feats.shape[1], proj_dim, generator=gen)
    z = feats.cpu() @ proj
    idx = int(torch.randint(feats.shape[0], (1,), generator=gen))
    selected = [idx]
    min_d = torch.cdist(z, z[idx : idx + 1]).squeeze(1)
    for _ in range(k - 1):
        idx = int(torch.argmax(min_d))
        selected.append(idx)
        min_d = torch.minimum(min_d, torch.cdist(z, z[idx : idx + 1]).squeeze(1))
    return torch.tensor(selected)


class PatchCore:
    """Memory bank of normal patches and nearest-neighbour anomaly maps."""

    def __init__(self, acfg: dict[str, Any], device: torch.device, seed: int) -> None:
        self.cfg, self.device, self.seed = acfg, device, seed
        self.extractor = PatchFeatures(acfg["pretrained"]).to(device)
        self.bank: torch.Tensor | None = None

    @property
    def pretrained(self) -> bool:
        """Whether ImageNet weights were available."""
        return self.extractor.trunk.pretrained

    def fit(self, batches_of_images) -> PatchCore:
        """Build the bank from an iterable of ``(B, 3, H, W)`` normal-image batches."""
        rng = torch.Generator().manual_seed(self.seed)
        kept = []
        for images in batches_of_images:
            feats = self.extractor(images.to(self.device))
            flat = feats.permute(0, 2, 3, 1).reshape(feats.shape[0], -1, feats.shape[1])
            for f in flat:
                take = torch.randperm(f.shape[0], generator=rng)[: self.cfg["patches_per_image"]]
                kept.append(f[take.to(f.device)])
        feats = torch.cat(kept)
        idx = greedy_coreset(feats, self.cfg["coreset"], self.cfg["projection_dim"], self.seed)
        self.bank = feats[idx.to(feats.device)].contiguous()
        return self

    @torch.no_grad()
    def score(self, images: torch.Tensor) -> np.ndarray:
        """Anomaly maps ``(B, H, W)`` at input resolution."""
        assert self.bank is not None, "fit() first"
        feats = self.extractor(images.to(self.device))
        b, c, h, w = feats.shape
        maps = []
        for f in feats.permute(0, 2, 3, 1).reshape(b, h * w, c):
            maps.append(torch.cdist(f, self.bank).min(dim=1).values.view(1, 1, h, w))
        up = (
            F.interpolate(
                torch.cat(maps), size=images.shape[-2:], mode="bilinear", align_corners=False
            )[:, 0]
            .cpu()
            .numpy()
        )
        sigma = self.cfg["blur_sigma"] * images.shape[-1] / 256
        return np.stack([cv2.GaussianBlur(m, (0, 0), sigma) for m in up])

    def state(self) -> dict[str, Any]:
        """Serialisable state (the bank) for saving."""
        return {"bank": self.bank.cpu() if self.bank is not None else None}

    def load_state(self, state: dict[str, Any]) -> PatchCore:
        """Restore a saved bank."""
        self.bank = state["bank"].to(self.device)
        return self
