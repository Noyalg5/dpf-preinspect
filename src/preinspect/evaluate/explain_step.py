"""Explanations for a run and their tests against the synthetic ground truth."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from preinspect.config import Config, data_dir
from preinspect.data.fusion_data import FusionDataset
from preinspect.data.io import decode_bitmask, load_latent, read_mask
from preinspect.explain import xai_tests as xt
from preinspect.explain.attributions import integrated_gradients, shap_values
from preinspect.explain.gradcam import gradcam
from preinspect.models.fusion_net import FusionNet
from preinspect.models.tabular import load_model
from preinspect.report.style import INK, SERIES, save_figure
from preinspect.utils.device import resolve_device

SCRAP = 2
VIEWS = ("inlet", "outlet")


def load_fusion(cfg: Config, run_path: Path, name: str, device: torch.device):
    """Rebuild a saved fusion network and its metadata preprocessor."""
    ckpt = torch.load(
        run_path / "models" / "fusion" / f"{name}.pt", map_location="cpu", weights_only=False
    )
    model = FusionNet(
        ckpt["prep"].dim, cfg["fusion"]["width"], tuple(ckpt["modalities"]), pretrained=False
    )
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval(), ckpt["prep"]


def _known_mask(cfg: Config, uid: str, view: str, size: int) -> np.ndarray:
    bits = [cfg["defect_classes"].index(c) for c in cfg["perception"]["classes"]]
    m = decode_bitmask(read_mask(data_dir(cfg), uid, view), len(cfg["defect_classes"]))[bits]
    return (
        cv2.resize(m.any(axis=0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        > 0
    )


def _randomised(model: FusionNet, layer: str) -> FusionNet:
    clone = copy.deepcopy(model)
    for module in getattr(clone.image, layer).modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return clone.eval()


def gradcam_tests(
    cfg: Config, model: FusionNet, prep, units: pd.DataFrame, device: torch.device
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Energy-in-mask, pointing game, randomisation check and deletion/insertion."""
    e, size = cfg["explain"], cfg["fusion"]["image_size"]
    ds = FusionDataset(data_dir(cfg), units, prep, cfg["generator"], size)
    rand_model = _randomised(model, e["gradcam_layer"])
    rows, examples, curves = [], [], {"cam_del": [], "cam_ins": [], "rnd_del": [], "rnd_ins": []}
    rng = np.random.default_rng(cfg["seed"])
    for i in range(len(ds)):
        item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
        cams = gradcam(model, item, SCRAP, e["gradcam_layer"])
        rcams = gradcam(rand_model, item, SCRAP, e["gradcam_layer"])
        uid = units.iloc[i]["unit_id"]
        best = None
        for v, view in enumerate(VIEWS):
            mask = _known_mask(cfg, uid, view, size)
            if not mask.any():
                continue
            cam = cams[v][0]
            rows.append(
                {
                    "unit_id": uid,
                    "view": view,
                    "energy_in_mask": xt.energy_in_mask(cam, mask),
                    "mask_area": float(mask.mean()),
                    "pointing_hit": xt.pointing_hit(cam, mask, e["pointing_tolerance_px"]),
                    "pointing_chance": xt.tolerance_area(mask, e["pointing_tolerance_px"]),
                    "randomised_similarity": xt.map_similarity(cam, rcams[v][0]),
                }
            )
            if best is None or mask.mean() > best[2].mean():
                best = (v, view, mask, cam)
        if best is None:
            continue
        if len(examples) < 6:
            examples.append(
                {
                    "image": item[best[1]][0].permute(1, 2, 0).cpu().numpy(),
                    "cam": best[3],
                    "mask": best[2],
                    "unit_id": uid,
                    "view": best[1],
                }
            )
        if i < e["n_deletion"]:
            v, view, _, cam = best
            img = item[view][0].permute(1, 2, 0).cpu().numpy()
            blurred = cv2.GaussianBlur(img, (0, 0), size / 32)

            @torch.no_grad()
            def score(x: np.ndarray, view: str = view, item: dict = item) -> float:
                b = dict(item)
                b[view] = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
                out = model(b["inlet"], b["outlet"], b["flow"], b["meta"])
                return float(torch.softmax(out["logits"], 1)[0, SCRAP])

            for tag, sal in (("cam", cam), ("rnd", rng.random(cam.shape))):
                d, ins = xt.deletion_insertion(score, img, blurred, sal, e["deletion_steps"])
                curves[f"{tag}_del"].append(d)
                curves[f"{tag}_ins"].append(ins)
    df = pd.DataFrame(rows)
    steps = e["deletion_steps"]
    mean_curve = {k: np.mean(v, axis=0).tolist() for k, v in curves.items() if v}
    summary = {
        "images": int(len(df)),
        "energy_in_mask": float(df["energy_in_mask"].mean()),
        "mask_area_share": float(df["mask_area"].mean()),
        "pointing_accuracy": float(df["pointing_hit"].mean()),
        "pointing_chance": float(df["pointing_chance"].mean()),
        "randomised_similarity": float(df["randomised_similarity"].mean()),
        "deletion_auc": {
            k: xt.curve_auc(mean_curve[f"{k}_del"], steps)
            for k in ("cam", "rnd")
            if f"{k}_del" in mean_curve
        },
        "insertion_auc": {
            k: xt.curve_auc(mean_curve[f"{k}_ins"], steps)
            for k in ("cam", "rnd")
            if f"{k}_ins" in mean_curve
        },
        "curves": {"steps": steps, **mean_curve},
    }
    return summary, examples


def ig_importance(
    cfg: Config, model: FusionNet, prep, units: pd.DataFrame, device: torch.device
) -> dict[str, Any]:
    """Mean absolute IG attribution per metadata field and flow point (SCRAP logit)."""
    ds = FusionDataset(data_dir(cfg), units, prep, cfg["generator"], cfg["fusion"]["image_size"])
    flow_parts, meta_parts = [], []
    for batch in DataLoader(ds, batch_size=8):
        b = {k: v.to(device) for k, v in batch.items()}
        fa, ma = integrated_gradients(model, b, SCRAP)
        flow_parts.append(np.abs(fa[:, 0]))
        meta_parts.append(np.abs(ma))
    meta = np.concatenate(meta_parts).mean(axis=0)
    names = [n for c in prep.numeric for n in (c, c)]
    names += [f"{col}={v}" for col, cats in prep.categories.items() for v in cats]
    per_field: dict[str, float] = {}
    for n, val in zip(names, meta, strict=True):
        key = n.split("=")[0]
        per_field[key] = per_field.get(key, 0.0) + float(val)
    flow = np.concatenate(flow_parts).mean(axis=0)
    return {
        "metadata": per_field,
        "flow_points": flow.tolist(),
        "flow_total": float(flow.sum()),
        "metadata_total": float(meta.sum()),
    }


def shap_tests(cfg: Config, run_path: Path, features: pd.DataFrame) -> dict[str, Any]:
    """SHAP on the deployed concept-route model and agreement with the true drivers."""
    e = cfg["explain"]
    model = load_model(run_path / "models" / "tabular" / f"{cfg['deploy']['model']}.joblib")
    train = features[features["split"] == "train"]
    test = features[features["split"] == "test"]
    bg = train.sample(min(e["shap_background"], len(train)), random_state=cfg["seed"])
    ex = test.sample(min(e["shap_units"], len(test)), random_state=cfg["seed"])
    values = shap_values(model, features, bg, ex, SCRAP, cfg["seed"])
    mean_abs = values.abs().mean().sort_values(ascending=False)
    prefixes = {
        "vision": ("lrn_", "anom_"),
        "flow": ("flow_",),
        "metadata": ("meta_",),
        "cross-modal": ("xmod_",),
    }
    per_unit = pd.DataFrame(
        {m: values[[c for c in values if c.startswith(p)]].sum(axis=1) for m, p in prefixes.items()}
    )
    per_unit.insert(0, "unit_id", ex["unit_id"].to_numpy())
    per_unit.to_parquet(run_path / "tables" / "modality_contribution.parquet", index=False)
    latent = load_latent(data_dir(cfg)).set_index("unit_id").loc[features["unit_id"]].reset_index()
    groups = e["driver_groups"]
    true_imp = xt.latent_group_importance(
        latent,
        (features["split"] == "train").to_numpy(),
        (features["split"] == "test").to_numpy(),
        groups,
        cfg["seed"],
    )
    model_imp = xt.model_group_importance(mean_abs, groups)
    return {
        "top_features": mean_abs.head(15).to_dict(),
        "modality_share": (
            per_unit.drop(columns="unit_id").abs().mean()
            / per_unit.drop(columns="unit_id").abs().mean().sum()
        ).to_dict(),
        "group_importance_model": model_imp,
        "group_importance_true": true_imp,
        "driver_rank_agreement": xt.rank_agreement(model_imp, true_imp),
    }


def figures(run_path: Path, gc: dict[str, Any], examples: list[dict], shap_res: dict) -> None:
    """Grad-CAM examples, global SHAP bars and deletion/insertion curves."""
    if examples:
        fig, axes = plt.subplots(
            2, len(examples), figsize=(2.2 * len(examples), 4.6), squeeze=False
        )
        for k, ex in enumerate(examples):
            axes[0, k].imshow(np.clip(ex["image"], 0, 1))
            axes[0, k].contour(ex["mask"], levels=[0.5], colors=[SERIES[1]], linewidths=0.8)
            axes[0, k].set_title(f"{ex['unit_id']}\n{ex['view']}", fontsize=7)
            axes[1, k].imshow(np.clip(ex["image"], 0, 1))
            axes[1, k].imshow(ex["cam"], cmap="magma", alpha=0.55)
            for ax in axes[:, k]:
                ax.axis("off")
        fig.suptitle(
            "Top: true defect outline. Bottom: Grad-CAM for SCRAP (synthetic data)", fontsize=9
        )
        save_figure(fig, run_path / "figures" / "gradcam_examples.png")
    top = pd.Series(shap_res["top_features"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(top.index, top.values, color=SERIES[0], height=0.6)
    ax.set_xlabel("Mean |SHAP| on P(SCRAP)")
    ax.set_title("Most influential features, concept route (test units)")
    save_figure(
        fig, run_path / "figures" / "shap_global.png", top.rename("mean_abs_shap").reset_index()
    )
    curves = gc["curves"]
    if "cam_del" in curves:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
        for ax, kind, title in ((axes[0], "del", "Deletion"), (axes[1], "ins", "Insertion")):
            ax.plot(curves["steps"], curves[f"cam_{kind}"], color=SERIES[0], label="Grad-CAM order")
            ax.plot(
                curves["steps"], curves[f"rnd_{kind}"], color=INK["muted"], label="random order"
            )
            ax.set_xlabel("Share of pixels changed")
            ax.set_ylabel("P(SCRAP)")
            ax.set_title(title)
            ax.legend()
        save_figure(
            fig,
            run_path / "figures" / "deletion_insertion.png",
            pd.DataFrame({k: v for k, v in curves.items()}),
        )


def explain_step(cfg: Config, run_path: Path) -> dict[str, Any]:
    """Run every explanation and its test; return the metrics."""
    device = resolve_device(cfg["device"])
    features = pd.read_parquet(run_path / "features" / "features.parquet")
    units = pd.read_parquet(data_dir(cfg) / "validation" / "units_clean.parquet")
    latent = load_latent(data_dir(cfg)).set_index("unit_id")
    test = units[units["split"] == "test"]
    scrap = test[latent.loc[test["unit_id"], "outcome_true"].to_numpy() == "SCRAP"]
    model, prep = load_fusion(cfg, run_path, "e2e_all_s0", device)
    gc, examples = gradcam_tests(cfg, model, prep, scrap.head(cfg["explain"]["n_units"]), device)
    ig = ig_importance(
        cfg,
        model,
        prep,
        test.sample(min(cfg["explain"]["ig_units"], len(test)), random_state=cfg["seed"]),
        device,
    )
    shap_res = shap_tests(cfg, run_path, features)
    figures(run_path, gc, examples, shap_res)
    return {"gradcam": gc, "integrated_gradients": ig, "shap": shap_res}
