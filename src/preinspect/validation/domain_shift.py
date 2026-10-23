"""Domain shift and the commissioning procedure for a new product line.

The shift set is a later product line with an unseen cell density imaged by
a different camera. Onboarding uses its first labelled units:

1. fine-tune the perception U-Nets on their defect masks (few epochs, low rate);
2. extend the anomaly memory banks with their defect-free faces;
3. keep the outcome model, calibration and routing thresholds unchanged,
   since they act on physical concepts rather than pixels.

The remaining shift units are scored before and after.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from preinspect.data.io import read_mask
from preinspect.data.torch_data import FaceDataset
from preinspect.models.perception import batches, class_bits, positive_weights
from preinspect.models.unet import segmentation_loss
from preinspect.models.vision import image_batch
from preinspect.service.pipeline import InspectionPipeline

VIEWS = ("inlet", "outlet")


def commission(
    cfg: dict, pipe: InspectionPipeline, ddir: Path, unit_ids: list[str]
) -> InspectionPipeline:
    """Return a copy of the pipeline adapted with the commissioning units."""
    d = cfg["campaign"]["domain_shift"]
    p = cfg["perception"]
    new = copy.copy(pipe)
    items = [(u, v) for u in unit_ids for v in VIEWS]
    bits = class_bits(cfg)
    new.unets = []
    for k, net in enumerate(pipe.unets):
        model = copy.deepcopy(net).train()
        ds = FaceDataset(
            ddir, items, bits, augment=p["augment"], crop=p["crop"], seed=cfg["seed"] + k
        )
        loader = DataLoader(
            ds,
            batch_size=p["batch_size"],
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(cfg["seed"] + k),
        )
        pos_weight = positive_weights(ddir, items, bits, p["pos_weight_max"]).to(pipe.device)
        opt = torch.optim.AdamW(model.parameters(), lr=d["finetune_lr"])
        for epoch in range(d["finetune_epochs"]):
            ds.epoch = epoch
            for img, masks, presence in loader:
                img, masks, presence = (t.to(pipe.device) for t in (img, masks, presence))
                seg, pres = model(img)
                loss = segmentation_loss(seg, pres, masks, presence, pos_weight, p["dice_weight"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        new.unets.append(model.eval())
    normal = [it for it in items if not read_mask(ddir, *it).any()]
    new.banks = []
    for bank in pipe.banks:
        extra = copy.copy(bank)
        extra.bank = None
        extra.fit(image_batch(ddir, b) for b in batches(normal, p["infer_batch"]))
        merged = copy.copy(bank)
        merged.bank = torch.cat([bank.bank, extra.bank])
        new.banks.append(merged)
    return new


def split_shift(units: pd.DataFrame, n_commission: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First units by arrival for commissioning; the rest for evaluation."""
    shift = units[units["split"] == "shift"].sort_values(["arrival_date", "unit_id"])
    return shift.head(n_commission), shift.iloc[n_commission:]
