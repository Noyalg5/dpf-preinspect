"""Attributions for non-image inputs: Integrated Gradients and SHAP.

- Integrated Gradients (Sundararajan et al., 2017) on the fusion network's
  flow curve and metadata vector, relative to a reference "new part, average
  unit": a flow curve equal to the new-part curve and metadata at the
  training mean.
- SHAP (Lundberg & Lee, 2017) on the boosting models via the model-agnostic
  permutation explainer, which supports scikit-learn's histogram boosting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
from captum.attr import IntegratedGradients

from preinspect.models.fusion_net import FusionNet


def integrated_gradients(
    model: FusionNet, batch: dict[str, torch.Tensor], target: int, steps: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Attributions to the flow curve ``(B, 2, n)`` and metadata ``(B, d)`` inputs.

    The image embedding is computed once and held fixed, so the integration
    path only varies flow and metadata.
    """
    with torch.no_grad():
        image = (
            model.embed_image(batch["inlet"], batch["outlet"])
            if "image" in model.modalities
            else None
        )

    def forward(flow: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # Captum stacks integration steps as whole-batch copies, so tile the embedding.
        emb = None if image is None else image.repeat(flow.shape[0] // image.shape[0], 1)
        return model.fuse(emb, flow, meta)["logits"]

    ig = IntegratedGradients(forward)
    flow_base = torch.ones_like(batch["flow"])
    meta_base = torch.zeros_like(batch["meta"])
    attr = ig.attribute(
        (batch["flow"], batch["meta"]),
        baselines=(flow_base, meta_base),
        target=target,
        n_steps=steps,
        internal_batch_size=batch["flow"].shape[0],
    )
    return attr[0].detach().cpu().numpy(), attr[1].detach().cpu().numpy()


def shap_values(
    model: Any,
    table: pd.DataFrame,
    background: pd.DataFrame,
    explain: pd.DataFrame,
    target: int,
    seed: int,
) -> pd.DataFrame:
    """Permutation SHAP values of P(target) for the rows in ``explain``.

    Categorical columns are passed to SHAP as integer codes and restored to
    their categories before prediction.
    """
    cols = model.columns
    cats = {
        c: table[c].cat.categories for c in cols if isinstance(table[c].dtype, pd.CategoricalDtype)
    }

    def encode(df: pd.DataFrame) -> np.ndarray:
        out = df[cols].copy()
        for c in cats:
            out[c] = out[c].cat.codes.replace(-1, np.nan)
        return out.to_numpy(dtype=float)

    def predict(x: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(x, columns=cols)
        for c, categories in cats.items():
            codes = frame[c].to_numpy()
            frame[c] = pd.Categorical.from_codes(
                np.nan_to_num(codes, nan=-1).astype(int), categories=categories
            )
        order = list(model.classifier.classes_)
        return model.classifier.predict_proba(frame)[
            :, order.index(["PASS", "REWORK", "SCRAP"][target])
        ]

    explainer = shap.PermutationExplainer(predict, encode(background), seed=seed)
    values = explainer(encode(explain), max_evals=2 * len(cols) + 1, silent=True).values
    return pd.DataFrame(values, columns=cols, index=explain.index)
