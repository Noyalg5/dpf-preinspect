"""Cross-fitted vision models and the pass that turns images into concepts.

Training units are split into folds. Each fold gets its own U-Net and its
own anomaly memory bank. A training unit is described by the models that
did *not* see it, so the downstream outcome model trains on concepts as
noisy as those it will meet in service; every other unit (and deployment)
uses the average of all folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from preinspect.classical.localise import localise_substrate
from preinspect.data.io import decode_bitmask, read_image, read_mask
from preinspect.evaluate.segmentation import AnomalyAccumulator, SegmentationAccumulator
from preinspect.features.learned import anomaly_concepts, learned_concepts
from preinspect.models.patchcore import PatchCore
from preinspect.models.perception import batches, class_bits, predict_with_embedding, train_unet
from preinspect.models.unet import UNet
from preinspect.runs import Run
from preinspect.utils.logging import get_logger

log = get_logger(__name__)
VIEWS = ("inlet", "outlet")


def image_batch(ddir: Path, items: list[tuple[str, str]]) -> torch.Tensor:
    """Load ``(unit, view)`` images as a ``[0, 1]`` float batch."""
    arr = np.stack([read_image(ddir, u, v) for u, v in items]).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())


def assign_folds(table: pd.DataFrame, k: int) -> dict[str, int]:
    """Deterministic fold for each training unit (alternating by unit ID)."""
    train_ids = sorted(table.loc[table["split"] == "train", "unit_id"])
    return {uid: i % k for i, uid in enumerate(train_ids)}


def fit_vision_models(
    cfg: dict, ddir: Path, table: pd.DataFrame, run: Run, device: torch.device
) -> tuple[list[UNet], list[PatchCore], dict[str, Any]]:
    """Train one U-Net and one anomaly bank per fold and save them in the run."""
    k = cfg["perception"]["folds"]
    folds = assign_folds(table, k)
    unets, banks, info = [], [], {"folds": k, "unet": [], "anomaly": []}
    for f in range(k):
        items = [(u, v) for u, fold in folds.items() if fold == f for v in VIEWS]
        unet, history = train_unet(cfg, ddir, items, device, cfg["seed"] + f)
        torch.save(unet.state_dict(), run.sub("models", "perception", f"unet_fold{f}.pt"))
        unets.append(unet)
        info["unet"].append({"items": len(items), **history})
        normal = [it for it in items if not read_mask(ddir, *it).any()]
        bank = PatchCore(cfg["anomaly"], device, cfg["seed"] + f)
        bank.fit(image_batch(ddir, b) for b in batches(normal, cfg["perception"]["infer_batch"]))
        torch.save(bank.state(), run.sub("models", "anomaly", f"bank_fold{f}.pt"))
        banks.append(bank)
        info["anomaly"].append(
            {
                "normal_images": len(normal),
                "bank_size": int(bank.bank.shape[0]),
                "pretrained": bank.pretrained,
            }
        )
        log.info(
            "fold %d: U-Net on %d images, bank from %d normal images", f, len(items), len(normal)
        )
    return unets, banks, info


def describe_arrays(
    cfg: dict,
    images: list[np.ndarray],
    lighting: list[str],
    diameters: list[float],
    unets: list[UNet],
    banks: list[PatchCore],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run perception and the anomaly guard on in-memory RGB images and measure concepts."""
    thr = cfg["perception"]["mask_threshold"]
    classes = cfg["perception"]["classes"]
    arr = np.stack(images).astype(np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())
    prob, presence, embedding = predict_with_embedding(unets, x, device)
    amaps = np.mean([bank.score(x) for bank in banks], axis=0)
    out = []
    for n, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        loc = localise_substrate(gray, cfg["classical"], lighting[n])
        lrn = learned_concepts(
            prob[n], presence[n], classes, loc, diameters[n] / (2 * loc.radius), thr
        )
        anom = anomaly_concepts(amaps[n], prob[n], thr, cfg["anomaly"]["explained_dilate_px"])
        out.append(
            {
                "lrn": lrn,
                "anom": anom,
                "prob": prob[n],
                "presence": presence[n],
                "amap": amaps[n],
                "loc": loc,
                "embedding": None if embedding is None else embedding[n],
            }
        )
    return out


def describe_images(
    cfg: dict,
    ddir: Path,
    items: list[tuple[str, str]],
    lighting: list[str],
    diameters: list[float],
    unets: list[UNet],
    banks: list[PatchCore],
    device: torch.device,
) -> list[dict[str, Any]]:
    """As :func:`describe_arrays`, reading ``(unit, view)`` images from the dataset."""
    images = [read_image(ddir, u, v) for u, v in items]
    out = describe_arrays(cfg, images, lighting, diameters, unets, banks, device)
    for rec, (uid, view) in zip(out, items, strict=True):
        rec.update(unit_id=uid, view=view)
    return out


def vision_pass(
    cfg: dict,
    ddir: Path,
    table: pd.DataFrame,
    unets: list[UNet],
    banks: list[PatchCore],
    device: torch.device,
) -> tuple[pd.DataFrame, dict, dict[str, np.ndarray]]:
    """Describe every unit's two faces and collect perception and guard metrics.

    Returns
    -------
    features : pandas.DataFrame
        One row per unit: ``lrn_<view>_<concept>``, ``anom_<view>_*``,
        ``anom_max`` and ``anom_unexplained``.
    metrics : dict
        Segmentation metrics on the test split and anomaly metrics on the
        test and novel-defect sets.
    """
    k = len(unets)
    folds = assign_folds(table, k)
    bits = class_bits(cfg)
    melt_bit = cfg["defect_classes"].index("melt")
    seg = SegmentationAccumulator(cfg["perception"]["classes"])
    anom_test, anom_novel = AnomalyAccumulator(), AnomalyAccumulator()
    rows: dict[str, dict[str, float]] = {}
    embeddings: dict[str, list[np.ndarray]] = {}
    meta = table.set_index("unit_id")
    groups: dict[tuple[int, ...], list[str]] = {}
    for uid in table["unit_id"]:
        use = tuple(i for i in range(k) if folds.get(uid) != i) if uid in folds else tuple(range(k))
        groups.setdefault(use, []).append(uid)
    for use, uids in groups.items():
        items = [(u, v) for u in uids for v in VIEWS]
        for chunk in batches(items, cfg["perception"]["infer_batch"]):
            lighting = [meta.at[u, f"lighting_{v}"] for u, v in chunk]
            diam = [float(meta.at[u, "meta_diameter_mm"]) for u, _ in chunk]
            for rec in describe_images(
                cfg,
                ddir,
                chunk,
                lighting,
                diam,
                [unets[i] for i in use],
                [banks[i] for i in use],
                device,
            ):
                uid, view = rec["unit_id"], rec["view"]
                row = rows.setdefault(uid, {})
                row.update({f"lrn_{view}_{c}": v for c, v in rec["lrn"].items()})
                row.update({f"anom_{view}_{c}": v for c, v in rec["anom"].items()})
                if rec.get("embedding") is not None:
                    embeddings.setdefault(uid, []).append(rec["embedding"])
                split = meta.at[uid, "split"]
                if split in ("test", "novel"):
                    truth = decode_bitmask(read_mask(ddir, uid, view), len(cfg["defect_classes"]))
                    if split == "test":
                        seg.update(
                            rec["prob"],
                            truth[bits],
                            rec["presence"],
                            cfg["perception"]["mask_threshold"],
                        )
                        anom_test.update(rec["amap"], truth.any(axis=0))
                        if not truth.any():
                            anom_novel.update(rec["amap"], truth[melt_bit])
                    elif truth[melt_bit].any():
                        anom_novel.update(rec["amap"], truth[melt_bit])
    features = pd.DataFrame.from_dict(rows, orient="index").rename_axis("unit_id").reset_index()
    for c in ("max", "unexplained"):
        features[f"anom_{c}"] = features[[f"anom_{v}_{c}" for v in VIEWS]].max(axis=1)
    metrics = {
        "segmentation_test": seg.summary(),
        "anomaly_test": anom_test.summary(),
        "anomaly_novel_vs_normal": anom_novel.summary(),
    }
    return features, metrics, {u: np.mean(e, axis=0) for u, e in embeddings.items()}
