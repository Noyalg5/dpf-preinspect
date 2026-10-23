"""Training and inference for the end-to-end fusion network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preinspect.data.fusion_data import OUTCOMES, FusionDataset, MetaPreprocessor
from preinspect.models.fusion_net import FusionNet
from preinspect.utils.logging import get_logger

log = get_logger(__name__)


def _to(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_fusion(
    cfg: dict[str, Any],
    ddir: Path,
    units: pd.DataFrame,
    modalities: list[str],
    seed: int,
    device: torch.device,
) -> tuple[FusionNet, MetaPreprocessor, dict]:
    """Train on the training split with a class-weighted multi-task loss.

    Loss = weighted cross-entropy (outcome) + L1 (ΔP ratio) + BCE (integrity).
    Class weights are inverse training frequencies, so SCRAP is not drowned
    out by the more common PASS units.
    """
    f = cfg["fusion"]
    torch.manual_seed(seed)
    train = units[units["split"] == "train"]
    prep = MetaPreprocessor().fit(train, cfg)
    ds = FusionDataset(
        ddir, train, prep, cfg["generator"], f["image_size"], cfg["perception"]["augment"], seed
    )
    loader = DataLoader(
        ds,
        batch_size=f["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = train["disposition"].value_counts()
    weights = torch.tensor(
        [len(train) / (3 * counts.get(c, 1)) for c in OUTCOMES], dtype=torch.float32, device=device
    )
    model = FusionNet(prep.dim, f["width"], tuple(modalities), f["pretrained"]).to(device)
    encoder = [p for n, p in model.named_parameters() if n.startswith("image.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("image.")]
    groups = [{"params": rest, "lr": f["lr"]}]
    if encoder:
        groups.append({"params": encoder, "lr": f["lr_encoder"]})
    opt = torch.optim.AdamW(groups, weight_decay=f["weight_decay"])
    steps = max(1, f["epochs"] * len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=steps
    )
    losses = []
    for epoch in range(f["epochs"]):
        ds.epoch = epoch
        model.train()
        running = []
        for batch in loader:
            b = _to(batch, device)
            out = model(b["inlet"], b["outlet"], b["flow"], b["meta"])
            ok = torch.isfinite(b["dp"])
            loss = (
                F.cross_entropy(out["logits"], b["outcome"], weight=weights)
                + f["dp_weight"] * F.l1_loss(out["dp"][ok], b["dp"][ok])
                + f["integrity_weight"]
                * F.binary_cross_entropy_with_logits(out["integrity"], b["integrity"])
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running.append(float(loss.detach()))
        losses.append(float(np.mean(running)) if running else float("nan"))
        log.info(
            "fusion %s seed %d epoch %d/%d loss %.4f",
            "+".join(modalities),
            seed,
            epoch + 1,
            f["epochs"],
            losses[-1],
        )
    pretrained = bool(getattr(getattr(model, "image", None), "pretrained", False))
    return model.eval(), prep, {"loss": losses, "pretrained": pretrained}


@torch.no_grad()
def predict_fusion(
    model: FusionNet,
    prep: MetaPreprocessor,
    cfg: dict[str, Any],
    ddir: Path,
    units: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    """Logits (``z_*``), probabilities (``p_*``), ΔP ratio, integrity and gate weights."""
    ds = FusionDataset(ddir, units, prep, cfg["generator"], cfg["fusion"]["image_size"])
    loader = DataLoader(ds, batch_size=cfg["fusion"]["batch_size"], shuffle=False)
    parts = []
    for batch in loader:
        b = _to(batch, device)
        out = model(b["inlet"], b["outlet"], b["flow"], b["meta"])
        parts.append({k: v.float().cpu().numpy() for k, v in out.items()})
    logits = np.concatenate([p["logits"] for p in parts])
    proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    res = units[["unit_id", "split"]].reset_index(drop=True).copy()
    for k, c in enumerate(OUTCOMES):
        res[f"z_{c}"] = logits[:, k]
        res[f"p_{c}"] = proba[:, k]
    res["dp_pred"] = np.concatenate([p["dp"] for p in parts])
    res["p_integrity"] = 1 / (1 + np.exp(-np.concatenate([p["integrity"] for p in parts])))
    gates = np.concatenate([p["gates"] for p in parts])
    for k, m in enumerate(model.modalities):
        res[f"gate_{m}"] = gates[:, k]
    return res
