"""Image acquisition: lighting, seating pose, occlusion and camera response.

Lighting modes
--------------
- ``diffuse``: dome light, reflectance only.
- ``low_angle``: ring light at grazing incidence; edges of the relief
  (cell walls, crack lips, chip edges) light up, channel depths go dark.
- ``backlight``: light source behind the substrate with weak front fill.
  A healthy wall-flow filter blocks it; cracks, through-channels from
  missing plugs and chips let it through.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from preinspect.synth.face import Layers, Pose

LIGHT_COLOUR = np.array([1.0, 0.97, 0.9], dtype=np.float32)


def sample_acquisition(
    gcfg: dict[str, Any], rng: np.random.Generator, camera: str, **overrides: Any
) -> dict[str, Any]:
    """Draw acquisition settings for one image.

    Parameters
    ----------
    gcfg : dict
        Generator configuration.
    rng : numpy.random.Generator
        Generator for this image.
    camera : str
        Camera profile name.
    **overrides
        Fixed values (e.g. ``lighting="backlight"``) used by DoE and robustness studies.

    Returns
    -------
    dict
        JSON-serialisable acquisition record.
    """
    a = gcfg["acquisition"]
    modes = list(a["lighting"])
    probs = np.array([a["lighting"][m] for m in modes])
    acq = {
        "camera": camera,
        "lighting": modes[rng.choice(len(modes), p=probs / probs.sum())],
        "exposure_ev": float(rng.uniform(*a["exposure_ev"])),
        "wb_gains": [float(g) for g in 1.0 + rng.normal(0, a["wb_gain_sd"], 3)],
        "blur_sigma": float(rng.uniform(*a["blur_sigma_px"])),
        "offset": [float(o) for o in rng.normal(0, a["offset_frac_sd"], 2)],
        "rotation_deg": float(rng.uniform(*a["rotation_deg"])),
        "scale": float(1.0 + rng.normal(0, a["scale_sd"])),
        "gradient": [
            float(rng.uniform(0, 2 * math.pi)),
            float(rng.uniform(*a["gradient_strength"])),
        ],
        "clamp": None,
        "reflection": None,
        "seed": int(rng.integers(0, 2**31)),
    }
    if rng.random() < a["clamp_prob"]:
        acq["clamp"] = {
            "angle": float(rng.uniform(0, 2 * math.pi)),
            "inner_r": float(rng.uniform(*a["clamp_inner_r"])),
            "half_width": float(rng.uniform(*a["clamp_half_width_rad"])),
        }
    if rng.random() < a["reflection_prob"]:
        rho = a["reflection_max_r"] * math.sqrt(rng.random())
        theta = rng.uniform(0, 2 * math.pi)
        acq["reflection"] = {
            "centre": [rho * math.cos(theta), rho * math.sin(theta)],
            "sigma": float(rng.uniform(*a["reflection_sigma"])),
            "strength": float(rng.uniform(*a["reflection_strength"])),
        }
    acq.update(overrides)
    return acq


def make_pose(acq: dict[str, Any], size: int, diameter_mm: float, fov_mm: float) -> Pose:
    """Substrate placement for a render of ``size`` pixels."""
    radius = 0.5 * diameter_mm / fov_mm * size * acq["scale"]
    return Pose(
        cx=size / 2 + acq["offset"][0] * size,
        cy=size / 2 + acq["offset"][1] * size,
        radius_px=radius,
        rotation=math.radians(acq["rotation_deg"]),
    )


def _illumination(size: int, pose: Pose, gradient: list[float]) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    angle, strength = gradient
    proj = ((xs - pose.cx) * math.cos(angle) + (ys - pose.cy) * math.sin(angle)) / (size / 2)
    return 1.0 + strength * proj


def shade(L: Layers, acq: dict[str, Any], gcfg: dict[str, Any], pose: Pose) -> np.ndarray:
    """Convert physical layers to scene radiance under the chosen lighting."""
    size = L.r.shape[0]
    illum = _illumination(size, pose, acq["gradient"])[..., None]
    rng = np.random.default_rng(acq["seed"])
    if acq["lighting"] == "backlight":
        fill = gcfg["acquisition"]["backlight_front_fill"]
        glow = 1.4 * cv2.GaussianBlur(L.transmit, (0, 0), 1.2 * size / 512) + 0.6 * L.transmit
        radiance = fill * L.albedo * illum + glow[..., None] * LIGHT_COLOUR
    elif acq["lighting"] == "low_angle":
        la = gcfg["acquisition"]["low_angle"]
        h = cv2.GaussianBlur(L.height, (0, 0), 0.7)
        gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.clip(la["sobel_scale"] * np.hypot(gx, gy), 0.0, 1.0)
        shading = la["base"] + la["edge_gain"] * edge + la["top"] * np.clip(L.height, 0.0, 1.0)
        radiance = L.albedo * (shading * illum[..., 0])[..., None]
    else:
        radiance = L.albedo * illum
    if acq["lighting"] != "backlight" and L.gloss.any():
        sparkle = (rng.random(L.gloss.shape) > 0.97).astype(np.float32)
        sparkle = cv2.GaussianBlur(sparkle, (0, 0), 1.0) * 4.0
        radiance = radiance + (0.5 * L.gloss * sparkle)[..., None]
    if acq["reflection"] is not None:
        ref = acq["reflection"]
        d2 = (L.u - ref["centre"][0]) ** 2 + (L.v - ref["centre"][1]) ** 2
        spot = ref["strength"] * np.exp(-0.5 * d2 / ref["sigma"] ** 2)
        if acq["lighting"] == "backlight":
            spot *= 0.3
        radiance = radiance + spot[..., None]
    return radiance.astype(np.float32)


def clamp_mask(L: Layers, acq: dict[str, Any]) -> np.ndarray:
    """Pixels hidden by a fixture clamp jaw (empty if no clamp in this shot)."""
    if acq["clamp"] is None:
        return np.zeros_like(L.r, dtype=bool)
    c = acq["clamp"]
    ang = np.arctan2(L.v, L.u)
    delta = np.abs(np.angle(np.exp(1j * (ang - c["angle"]))))
    return (L.r > c["inner_r"]) & (delta < c["half_width"])


def camera_response(
    radiance: np.ndarray,
    occluded: np.ndarray,
    acq: dict[str, Any],
    gcfg: dict[str, Any],
    pose: Pose,
    out_size: int,
) -> np.ndarray:
    """Apply exposure, white balance, vignetting, resampling, blur, noise and gamma.

    Returns
    -------
    numpy.ndarray
        8-bit RGB image of shape ``(out_size, out_size, 3)``.
    """
    cam = gcfg["acquisition"]["cameras"][acq["camera"]]
    rng = np.random.default_rng(acq["seed"] + 1)
    clamp_colour = np.asarray(gcfg["appearance"]["colours"]["clamp"], dtype=np.float32)
    radiance = radiance.copy()
    radiance[occluded] = clamp_colour
    x = radiance * (2.0 ** acq["exposure_ev"])
    x = x * np.asarray(acq["wb_gains"], np.float32) * np.asarray(cam["tint"], np.float32)
    lum = x.mean(axis=2, keepdims=True)
    x = lum + cam["saturation"] * (x - lum)
    size = x.shape[0]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    d2 = ((xs - size / 2) ** 2 + (ys - size / 2) ** 2) / (size / 2) ** 2
    x = x * (1.0 - cam["vignette"] * d2)[..., None]
    x = cv2.resize(x, (out_size, out_size), interpolation=cv2.INTER_AREA)
    if acq["blur_sigma"] > 0.05:
        x = cv2.GaussianBlur(x, (0, 0), acq["blur_sigma"])
    sigma = np.sqrt(cam["noise_sd"] ** 2 * np.clip(x, 0, None) + cam["read_sd"] ** 2)
    x = x + rng.normal(0, 1, x.shape).astype(np.float32) * sigma
    x = np.clip(x, 0.0, 1.0) ** (1.0 / cam["gamma"])
    return np.round(x * 255).astype(np.uint8)
