"""End-to-end multi-modal network with gated intermediate fusion.

Each modality is encoded to a vector of the same width; a softmax gate,
computed from all embeddings, weights them per unit before the shared heads.
The gate weights double as a per-unit statement of how much each modality
contributed to the prediction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from preinspect.models.backbone import Trunk, normalise

MODALITIES = ("image", "flow", "metadata")


class FlowEncoder(nn.Module):
    """1D convolutions over the flow curve (normalised ΔP and a validity channel)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, 3, padding=1), nn.ReLU(), nn.Conv1d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.proj = nn.Linear(64, width)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        """``curve``: ``(B, 2, n_points)``."""
        h = self.conv(curve)
        return self.proj(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))


class FusionNet(nn.Module):
    """Outcome, ΔP-ratio and integrity prediction from any subset of modalities.

    Parameters
    ----------
    meta_dim : int
        Length of the preprocessed metadata vector.
    width : int
        Embedding width per modality.
    modalities : tuple of str
        Modalities this network uses.
    pretrained : bool
        ImageNet initialisation for the image encoder when available.
    """

    def __init__(
        self,
        meta_dim: int,
        width: int = 128,
        modalities: tuple[str, ...] = MODALITIES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.modalities = tuple(m for m in MODALITIES if m in modalities)
        if "image" in self.modalities:
            self.image = Trunk(pretrained)
            self.image_proj = nn.Sequential(nn.Linear(1024, width), nn.ReLU())
        if "flow" in self.modalities:
            self.flow = FlowEncoder(width)
        if "metadata" in self.modalities:
            self.meta = nn.Sequential(nn.Linear(meta_dim, 64), nn.ReLU(), nn.Linear(64, width))
        m = len(self.modalities)
        self.gate = nn.Linear(width * m, m)
        self.trunk = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(0.2))
        self.outcome_head = nn.Linear(width, 3)
        self.dp_head = nn.Linear(width, 1)
        self.integrity_head = nn.Linear(width, 1)

    def embed_image(self, inlet: torch.Tensor, outlet: torch.Tensor) -> torch.Tensor:
        """Shared encoder on both faces, pooled and concatenated."""
        both = torch.cat([inlet, outlet], dim=0)
        pooled = self.image(normalise(both))[-1].mean(dim=(2, 3))
        n = inlet.shape[0]
        return self.image_proj(torch.cat([pooled[:n], pooled[n:]], dim=1))

    def forward(
        self, inlet: torch.Tensor, outlet: torch.Tensor, flow: torch.Tensor, meta: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return logits, ΔP ratio, integrity logit and gate weights."""
        image = self.embed_image(inlet, outlet) if "image" in self.modalities else None
        return self.fuse(image, flow, meta)

    def fuse(
        self, image: torch.Tensor | None, flow: torch.Tensor, meta: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Heads applied to a precomputed image embedding plus the raw flow and metadata.

        Separating this step lets attribution methods on flow and metadata run
        without re-encoding the images at every integration step.
        """
        emb = []
        if image is not None:
            emb.append(image)
        if "flow" in self.modalities:
            emb.append(self.flow(flow))
        if "metadata" in self.modalities:
            emb.append(self.meta(meta))
        stacked = torch.stack(emb, dim=1)
        gates = F.softmax(self.gate(torch.cat(emb, dim=1)), dim=1)
        fused = self.trunk((gates.unsqueeze(-1) * stacked).sum(dim=1))
        return {
            "logits": self.outcome_head(fused),
            "dp": self.dp_head(fused).squeeze(1),
            "integrity": self.integrity_head(fused).squeeze(1),
            "gates": gates,
        }
