"""ResNet-18 backbone with optional ImageNet weights.

Weights are used when cached or downloadable, unless
``PREINSPECT_OFFLINE_WEIGHTS=1``; otherwise the network starts from random
initialisation. Callers record which happened.
"""

from __future__ import annotations

import os

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from preinspect.utils.logging import get_logger

log = get_logger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_resnet18(pretrained: bool = True) -> tuple[nn.Module, bool]:
    """Return a ResNet-18 and whether ImageNet weights were loaded."""
    if pretrained and os.environ.get("PREINSPECT_OFFLINE_WEIGHTS") != "1":
        try:
            return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), True
        except Exception as exc:  # noqa: BLE001 - any download failure falls back
            log.warning("ImageNet weights unavailable (%s); using random init", exc)
    return resnet18(weights=None), False


class Trunk(nn.Module):
    """ResNet-18 split into stages so skips and intermediate maps are accessible."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        net, self.pretrained = load_resnet18(pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Feature maps at strides 2, 4, 8, 16 and 32."""
        s2 = self.stem(x)
        s4 = self.layer1(self.pool(s2))
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)
        return [s2, s4, s8, s16, s32]


def normalise(x: torch.Tensor) -> torch.Tensor:
    """ImageNet normalisation of a ``[0, 1]`` image batch."""
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
