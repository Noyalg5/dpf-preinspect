"""Data-validation gate catches every injected fault and refuses tampered data."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from preinspect.config import data_dir
from preinspect.data.validate import DataIntegrityError, require_valid, validate_dataset

FAULT_TO_CHECK = {
    "missing_value": "MISSING_REQUIRED",
    "out_of_range": "OUT_OF_RANGE",
    "duplicate_id": "DUPLICATE_ID",
    "unit_mixup_f": "PROBABLE_FAHRENHEIT",
    "label_flip": "DISPOSITION_MISMATCH",
}


def test_every_injected_fault_is_caught(tiny_data) -> None:
    ddir = data_dir(tiny_data)
    faults = pd.read_parquet(ddir / "faults.parquet")
    assert set(faults["fault_type"]) == set(FAULT_TO_CHECK)
    quarantine = pd.read_csv(ddir / "validation" / "quarantine.csv")
    for _, fault in faults.iterrows():
        checks = set(quarantine.loc[quarantine["unit_id"] == fault["unit_id"], "check"])
        assert FAULT_TO_CHECK[fault["fault_type"]] in checks, fault.to_dict()


def test_flow_dropouts_and_field_blanking(tiny_data) -> None:
    ddir = data_dir(tiny_data)
    units = pd.read_parquet(ddir / "units.parquet")
    signals = np.load(ddir / "signals.npy")
    n_nan = np.isnan(signals[units["signal_index"], 1]).sum(axis=1)
    heavy = set(units.loc[n_nan > tiny_data["validation"]["max_flow_dropouts"], "unit_id"])
    clean = require_valid(tiny_data)
    assert heavy.isdisjoint(clean["unit_id"])
    mixups = pd.read_parquet(ddir / "faults.parquet").query("fault_type == 'unit_mixup_f'")
    kept = clean[clean["unit_id"].isin(mixups["unit_id"])]
    assert kept["max_regen_temp_c"].isna().all()
    assert not clean["unit_id"].duplicated().any()


def test_tampered_file_blocks_training(tiny_data, tmp_path) -> None:
    cfg = {**tiny_data, "paths": {**tiny_data["paths"], "data_root": str(tmp_path)}}
    shutil.copytree(data_dir(tiny_data), data_dir(cfg))
    assert len(require_valid(cfg)) > 0
    image = next((data_dir(cfg) / "images").glob("*.jpg"))
    image.write_bytes(image.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(DataIntegrityError):
        require_valid(cfg)
    result = validate_dataset(cfg, write=False)
    assert not result.integrity_ok
    assert any(i["issue"] == "hash mismatch" for i in result.file_issues)
