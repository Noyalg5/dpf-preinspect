"""Latency of the full inspection path, and ONNX export of the perception model.

Timed per unit from compressed JPEG bytes to a routed decision with its
rationale, as the station would call it. CPU timings use ONNX Runtime for
the U-Nets when ``deploy.cpu_runtime`` is ``onnx``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch

from preinspect.models.unet import UNet
from preinspect.service.pipeline import InspectionPipeline, UnitInput

VIEWS = ("inlet", "outlet")


class OnnxUNet:
    """ONNX Runtime stand-in for a U-Net, callable like the torch module."""

    def __init__(self, path: Path) -> None:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seg, presence = self.session.run(None, {"image": x.detach().cpu().numpy()})
        return torch.from_numpy(seg), torch.from_numpy(presence)

    def eval(self) -> OnnxUNet:
        """Match the torch interface."""
        return self


def export_onnx(net: UNet, path: Path, size: int) -> Path:
    """Export a U-Net to ONNX with a dynamic batch dimension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, size, size)
    torch.onnx.export(
        net.cpu().eval(),
        dummy,
        str(path),
        input_names=["image"],
        output_names=["seg", "presence"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"image": {0: "batch"}, "seg": {0: "batch"}, "presence": {0: "batch"}},
    )
    return path


def onnx_unets(pipe: InspectionPipeline, folder: Path) -> list[OnnxUNet]:
    """Export each of the pipeline's U-Nets and load them in ONNX Runtime."""
    size = pipe.cfg["generator"]["image_size"]
    out = []
    for k, net in enumerate(pipe.unets):
        cpu_net = UNet(len(pipe.cfg["perception"]["classes"]), pretrained=False)
        cpu_net.load_state_dict({n: t.cpu() for n, t in net.state_dict().items()})
        out.append(OnnxUNet(export_onnx(cpu_net, folder / f"unet_fold{k}.onnx", size)))
    return out


def onnx_agreement(pipe: InspectionPipeline, onnx_nets: list[OnnxUNet], size: int) -> float:
    """Largest absolute difference between torch and ONNX outputs on a random image."""
    x = torch.rand(1, 3, size, size)
    torch_net = UNet(len(pipe.cfg["perception"]["classes"]), pretrained=False)
    torch_net.load_state_dict({n: t.cpu() for n, t in pipe.unets[0].state_dict().items()})
    with torch.no_grad():
        ref = torch.sigmoid(torch_net.eval()(x)[0]).numpy()
    got = torch.sigmoid(onnx_nets[0](x)[0]).numpy()
    return float(np.abs(ref - got).max())


def _encode(unit: UnitInput) -> dict[str, bytes]:
    return {
        v: cv2.imencode(".jpg", cv2.cvtColor(unit.images[v], cv2.COLOR_RGB2BGR))[1].tobytes()
        for v in VIEWS
    }


def time_pipeline(
    pipe: InspectionPipeline, units: list[UnitInput], warmup: int = 2
) -> dict[str, Any]:
    """Per-unit wall-clock latency (decode, inference, decision, rationale)."""
    encoded = [_encode(u) for u in units]
    times = []
    for k, (unit, jpgs) in enumerate(zip(units, encoded, strict=True)):
        start = time.perf_counter()
        images = {
            v: cv2.cvtColor(
                cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            for v, b in jpgs.items()
        }
        unit.images = images
        pipe.inspect(unit, return_maps=False)
        if k >= warmup:
            times.append(time.perf_counter() - start)
    arr = np.asarray(times)
    return {
        "p50_s": float(np.percentile(arr, 50)),
        "p95_s": float(np.percentile(arr, 95)),
        "mean_s": float(arr.mean()),
        "n": int(arr.size),
    }
