"""Inputs for the end-to-end fusion network: both faces, flow curve, metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preinspect.data.io import load_signals, read_image
from preinspect.data.torch_data import geometric, photometric
from preinspect.features.flow import normalised_curve
from preinspect.features.metadata import NUMERIC

OUTCOMES = ["PASS", "REWORK", "SCRAP"]


@dataclass
class MetaPreprocessor:
    """Standardise numeric metadata, add missing indicators, one-hot categoricals.

    Statistics are fitted on training units only.
    """

    numeric: list[str] = field(default_factory=lambda: list(NUMERIC))
    categories: dict[str, list[str]] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    def fit(self, units: pd.DataFrame, cfg: dict[str, Any]) -> MetaPreprocessor:
        """Fit means, standard deviations and category lists."""
        for col in self.numeric:
            vals = pd.to_numeric(units[col], errors="coerce")
            self.mean[col] = float(vals.mean())
            self.std[col] = float(vals.std()) or 1.0
        self.categories = {
            "material": list(cfg["generator"]["substrate"]["materials"]),
            "application": list(cfg["generator"]["applications"]),
        }
        return self

    @property
    def dim(self) -> int:
        """Length of the output vector."""
        return 2 * len(self.numeric) + sum(len(v) for v in self.categories.values())

    def transform(self, row: pd.Series) -> np.ndarray:
        """Vector for one unit; missing numeric values become 0 with indicator 1."""
        out = []
        for col in self.numeric:
            v = pd.to_numeric(row[col], errors="coerce")
            missing = pd.isna(v)
            out += [0.0 if missing else (float(v) - self.mean[col]) / self.std[col], float(missing)]
        for col, cats in self.categories.items():
            out += [float(row[col] == c) for c in cats]
        return np.asarray(out, dtype=np.float32)


class FusionDataset(Dataset):
    """One item per unit with targets from the recorded downstream tests."""

    def __init__(
        self,
        ddir: Path,
        units: pd.DataFrame,
        prep: MetaPreprocessor,
        gcfg: dict[str, Any],
        image_size: int,
        augment: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.ddir, self.units, self.prep = Path(ddir), units.reset_index(drop=True), prep
        self.gcfg, self.size, self.augment, self.seed = gcfg, image_size, augment, seed
        self.signals = load_signals(self.ddir)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.units)

    def _image(self, uid: str, view: str, rng: np.random.Generator | None) -> torch.Tensor:
        img = read_image(self.ddir, uid, view)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        if rng is not None:
            x = photometric(x, rng, self.augment)
            x, _ = geometric(x, np.zeros((1, 1, 1), np.float32), rng)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.units.iloc[idx]
        rng = (
            np.random.default_rng((self.seed, self.epoch, idx))
            if self.augment is not None
            else None
        )
        curve = normalised_curve(self.gcfg, row.to_dict(), self.signals[int(row["signal_index"])])
        dp = row.get("dp_ratio_post_clean", np.nan)
        return {
            "inlet": self._image(row["unit_id"], "inlet", rng),
            "outlet": self._image(row["unit_id"], "outlet", rng),
            "flow": torch.from_numpy(curve),
            "meta": torch.from_numpy(self.prep.transform(row)),
            "outcome": torch.tensor(
                OUTCOMES.index(row["disposition"]) if row.get("disposition") in OUTCOMES else -1
            ),
            "dp": torch.tensor(float(dp) if pd.notna(dp) else float("nan")),
            "integrity": torch.tensor(float(bool(row.get("integrity_pass", True)))),
        }
