"""Grad-CAM for the fusion network's image encoder (Selvaraju et al., 2017).

The class score's gradient with respect to a convolutional layer's
activations, averaged over space, weights each channel; the weighted sum of
activations, rectified and upsampled, is the heatmap.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from preinspect.models.fusion_net import FusionNet


def gradcam(
    model: FusionNet, batch: dict[str, torch.Tensor], target: int, layer: str = "layer4"
) -> tuple[np.ndarray, np.ndarray]:
    """Heatmaps for both faces of each unit in a batch.

    Parameters
    ----------
    model : FusionNet
        Network with an image branch.
    batch : dict
        Tensors ``inlet``, ``outlet``, ``flow``, ``meta`` on the model's device.
    target : int
        Class index to explain (2 = SCRAP).
    layer : str
        Encoder stage to read activations from.

    Returns
    -------
    inlet_cam, outlet_cam : numpy.ndarray
        ``(B, H, W)`` maps scaled to [0, 1] per image.
    """
    store: dict[str, torch.Tensor] = {}
    module = getattr(model.image, layer)

    def hook(_m, _inp, out):
        out.retain_grad()
        store["act"] = out

    handle = module.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        out = model(batch["inlet"], batch["outlet"], batch["flow"], batch["meta"])
        out["logits"][:, target].sum().backward()
    finally:
        handle.remove()
    act, grad = store["act"], store["act"].grad
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=batch["inlet"].shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[:, 0].detach().float().cpu().numpy()
    peak = cam.reshape(cam.shape[0], -1).max(axis=1)[:, None, None]
    cam = cam / np.where(peak > 0, peak, 1.0)
    n = batch["inlet"].shape[0]
    return cam[:n], cam[n:]
