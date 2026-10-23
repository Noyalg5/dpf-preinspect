"""U-Net with a ResNet-18 encoder and a multi-label image-level head (Ronneberger et al., 2015)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from preinspect.models.backbone import Trunk, normalise


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    """Upsample by two, concatenate the skip connection, two 3x3 convolutions."""

    def __init__(self, cin: int, cskip: int, cout: int) -> None:
        super().__init__()
        self.body = nn.Sequential(_conv(cin + cskip, cout), _conv(cout, cout))

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        size = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.body(x)


class UNet(nn.Module):
    """Segmentation of defect classes plus image-level presence logits.

    Parameters
    ----------
    n_classes : int
        Number of defect classes (one sigmoid output each; classes may overlap).
    pretrained : bool
        Start the encoder from ImageNet weights when available.
    """

    def __init__(self, n_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = Trunk(pretrained)
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.up0 = UpBlock(32, 0, 16)
        self.seg_head = nn.Conv2d(16, n_classes, 1)
        self.cls_head = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return segmentation logits ``(B, C, H, W)`` and presence logits ``(B, C)``."""
        s2, s4, s8, s16, s32 = self.encoder(normalise(x))
        d = self.up4(s32, s16)
        d = self.up3(d, s8)
        d = self.up2(d, s4)
        d = self.up1(d, s2)
        d = self.up0(d, None)
        pooled = s32.mean(dim=(2, 3))
        # Kept for drift monitoring; not an output, so the ONNX graph is unchanged.
        self.pooled = pooled.detach()
        presence = self.cls_head(pooled)
        return self.seg_head(d), presence


def segmentation_loss(
    seg_logits: torch.Tensor,
    presence_logits: torch.Tensor,
    masks: torch.Tensor,
    presence: torch.Tensor,
    pos_weight: torch.Tensor,
    dice_weight: float,
) -> torch.Tensor:
    """Pixel BCE + soft Dice per class + image-level BCE.

    Dice keeps thin, rare classes (cracks, missing plugs) from being swamped
    by background pixels; ``pos_weight`` up-weights positive pixels per class.
    """
    bce = F.binary_cross_entropy_with_logits(
        seg_logits, masks, pos_weight=pos_weight.view(1, -1, 1, 1)
    )
    prob = torch.sigmoid(seg_logits)
    inter = (prob * masks).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))
    dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
    cls = F.binary_cross_entropy_with_logits(presence_logits, presence)
    return bce + dice_weight * dice.mean() + cls
