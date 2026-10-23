"""Cell lattice estimation and periodic-pattern removal.

The plugged/open checkerboard is periodic, so its spectrum is dominated by
two diagonal peaks at ``|f| = 1 / (sqrt(2) * pitch)``, 45 degrees from the
cell axes. That gives pitch and orientation; a template search gives the
grid phase and which checkerboard parity is plugged.

Removing the spectral peaks of the lattice (a notch filter) leaves only the
non-periodic content of the face: cracks, stains, deposits and debris.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from preinspect.classical.localise import Localisation


@dataclass
class Lattice:
    """Cell grid in pixel units, relative to the substrate centre."""

    pitch: float
    angle: float
    offset: tuple[float, float]
    plugged_parity: int
    strength: float

    def cell_coords(self, xs: np.ndarray, ys: np.ndarray, loc: Localisation):
        """Cell indices and in-cell fractions for pixel coordinates."""
        dx, dy = xs - loc.cx, ys - loc.cy
        c, s = math.cos(self.angle), math.sin(self.angle)
        xp = (c * dx + s * dy - self.offset[0]) / self.pitch
        yp = (-s * dx + c * dy - self.offset[1]) / self.pitch
        i, j = np.floor(xp), np.floor(yp)
        return i.astype(int), j.astype(int), xp - i, yp - j

    def cell_centre(self, i: np.ndarray, j: np.ndarray, loc: Localisation):
        """Pixel coordinates of cell centres."""
        xp = self.offset[0] + (i + 0.5) * self.pitch
        yp = self.offset[1] + (j + 0.5) * self.pitch
        c, s = math.cos(self.angle), math.sin(self.angle)
        return loc.cx + c * xp - s * yp, loc.cy + s * xp + c * yp


def _crop(gray: np.ndarray, loc: Localisation, frac: float) -> tuple[np.ndarray, int, int]:
    half = int(loc.radius * frac / math.sqrt(2))
    x0, y0 = int(loc.cx) - half, int(loc.cy) - half
    x0, y0 = max(0, x0), max(0, y0)
    patch = gray[y0 : y0 + 2 * half, x0 : x0 + 2 * half].astype(np.float32)
    return patch, x0, y0


def expected_pitch_px(
    ccfg: dict[str, Any], cpsi: float, diameter_mm: float, radius_px: float
) -> float:
    """Pitch implied by the part's datasheet cell density, in pixels.

    A real filter has ``25.4 / sqrt(cpsi)`` mm pitch. The simulator renders a
    stylised face with fewer, larger cells, so the same relationship is scaled
    by the documented stylisation constant.
    """
    cells = (
        ccfg["lattice"]["cells_per_sqrt_cpsi"]
        * math.sqrt(cpsi)
        * diameter_mm
        / ccfg["lattice"]["reference_diameter_mm"]
    )
    return 2.0 * radius_px / cells


def estimate_lattice(
    gray: np.ndarray, loc: Localisation, ccfg: dict[str, Any], expected_pitch: float | None = None
) -> Lattice:
    """Estimate pitch, orientation, phase and plugged parity of the cell grid.

    Parameters
    ----------
    gray : numpy.ndarray
        Grey-level image.
    loc : Localisation
        Substrate circle.
    ccfg : dict
        ``classical`` configuration section.
    expected_pitch : float, optional
        Pitch implied by part metadata. When given, it decides between the
        checkerboard and wall-grid readings of the spectral peak; otherwise
        the reading with the stronger template correlation wins.

    Returns
    -------
    Lattice
        Estimated grid; ``strength`` is the template correlation (0 to 1).
    """
    cfg = ccfg["lattice"]
    patch, _, _ = _crop(gray, loc, cfg["crop_r"])
    patch = (patch - patch.mean()) * np.outer(
        np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
    )
    # Zero-padding plus parabolic peak interpolation gives sub-bin frequency
    # resolution; a 1 degree error already shifts edge cells by a third of a cell.
    size = int(2 ** np.ceil(np.log2(patch.shape[0] * cfg["zero_pad"])))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(patch, s=(size, size))))
    freqs = np.fft.fftshift(np.fft.fftfreq(size))
    fy, fx = np.meshgrid(freqs, freqs, indexing="ij")
    freq = np.hypot(fx, fy)
    n_lo, n_hi = cfg["cells_range"]
    f_lo = n_lo / (2 * math.sqrt(2) * loc.radius)
    f_hi = n_hi / (2 * math.sqrt(2) * loc.radius)
    band = (freq >= f_lo) & (freq <= f_hi * math.sqrt(2)) & (fy >= 0)
    ky, kx = np.unravel_index(int(np.argmax(np.where(band, spec, 0))), spec.shape)
    py = _parabolic(spec[ky - 1 : ky + 2, kx]) if 0 < ky < size - 1 else 0.0
    px = _parabolic(spec[ky, kx - 1 : kx + 2]) if 0 < kx < size - 1 else 0.0
    step = freqs[1] - freqs[0]
    peak_y, peak_x = fy[ky, kx] + py * step, fx[ky, kx] + px * step
    f, phi = max(math.hypot(peak_x, peak_y), 1e-6), math.atan2(peak_y, peak_x)
    # The strongest peak is usually the checkerboard diagonal; under low-angle
    # light the bright wall grid can dominate instead. Test both readings.
    hypotheses = [
        (1.0 / (math.sqrt(2) * f), (phi - math.pi / 4) % (math.pi / 2)),
        (1.0 / f, phi % (math.pi / 2)),
    ]
    plausible = [h for h in hypotheses if 2 * loc.radius / n_hi <= h[0] <= 2 * loc.radius / n_lo]
    if expected_pitch is not None and plausible:
        plausible = [min(plausible, key=lambda h: abs(math.log(h[0] / expected_pitch)))]
    best: Lattice | None = None
    for pitch, angle in plausible:
        offset, parity, strength = _phase_search(gray, loc, pitch, angle, cfg)
        if best is None or strength > best.strength:
            best = Lattice(pitch, angle, offset, parity, strength)
    if best is None:
        pitch, angle = hypotheses[0]
        best = Lattice(pitch, angle, (0.0, 0.0), 0, 0.0)
    return best


def _parabolic(v: np.ndarray) -> float:
    """Sub-sample offset of a peak from three samples around it."""
    denom = v[0] - 2 * v[1] + v[2]
    return float(0.5 * (v[0] - v[2]) / denom) if denom != 0 else 0.0


def _phase_search(gray, loc, pitch, angle, cfg) -> tuple[tuple[float, float], int, float]:
    """Find the grid offset maximising correlation with a checkerboard template."""
    patch, x0, y0 = _crop(gray, loc, cfg["crop_r"])
    ys, xs = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]].astype(np.float32) + 0.5
    dx, dy = xs + x0 - loc.cx, ys + y0 - loc.cy
    c, s = math.cos(angle), math.sin(angle)
    xr, yr = c * dx + s * dy, -s * dx + c * dy
    values = (patch - patch.mean()).ravel()
    best = (0.0, (0.0, 0.0), 0)
    steps = np.arange(cfg["phase_steps"]) * pitch / cfg["phase_steps"]
    for ox in steps:
        for oy in steps:
            u, v = (xr - ox) / pitch, (yr - oy) / pitch
            fu, fv = u - np.floor(u), v - np.floor(v)
            interior = (np.abs(fu - 0.5) < 0.3) & (np.abs(fv - 0.5) < 0.3)
            sign = np.where((np.floor(u) + np.floor(v)) % 2 == 0, 1.0, -1.0) * interior
            score = float(values @ sign.ravel()) / max(1, interior.sum())
            if abs(score) > abs(best[0]):
                best = (score, (float(ox), float(oy)), 0 if score > 0 else 1)
    score, offset, light_parity = best
    strength = abs(score) / (float(np.std(values)) + 1e-6)
    return offset, light_parity, float(min(1.0, strength))


def remove_lattice(gray: np.ndarray, face: np.ndarray, ccfg: dict[str, Any]) -> np.ndarray:
    """Suppress periodic spectral peaks, leaving non-periodic face content.

    Parameters
    ----------
    gray : numpy.ndarray
        Grey-level image (float).
    face : numpy.ndarray of bool
        Face pixels; the rest is filled with the face mean before filtering.
    ccfg : dict
        ``classical`` configuration section.

    Returns
    -------
    numpy.ndarray
        Residual image (same shape), zero-mean on the face.
    """
    cfg = ccfg["notch"]
    g = gray.astype(np.float32)
    mean = float(g[face].mean()) if face.any() else float(g.mean())
    filled = np.where(face, g, mean) - mean
    spec = np.fft.fftshift(np.fft.fft2(filled))
    logmag = np.log1p(np.abs(spec))
    background = median_filter(logmag, size=cfg["median_size"])
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dc = np.hypot(yy - h // 2, xx - w // 2) <= cfg["dc_radius"]
    peaks = (logmag - background > cfg["peak_threshold"]) & ~dc
    spec[peaks] = spec[peaks] / np.abs(spec[peaks]) * np.expm1(background[peaks])
    residual = np.real(np.fft.ifft2(np.fft.ifftshift(spec)))
    return np.where(face, residual, 0.0).astype(np.float32)
