"""Configuration, seeding, hashing and run-folder behaviour."""

from __future__ import annotations

import json

import numpy as np

from preinspect.config import deep_merge, load_aux, load_config
from preinspect.runs import create_run, latest_run
from preinspect.utils.hashing import sha256_bytes, sha256_json
from preinspect.utils.seeding import derive_seed, make_rng


def test_profiles_extend_base() -> None:
    base = load_config("base")
    tiny = load_config("tiny")
    assert tiny["name"] == "tiny"
    assert tiny["seed"] == base["seed"]
    assert tiny["generator"]["version"] == base["generator"]["version"]


def test_deep_merge_does_not_mutate() -> None:
    a = {"x": {"y": 1, "z": 2}}
    merged = deep_merge(a, {"x": {"y": 5}})
    assert merged == {"x": {"y": 5, "z": 2}}
    assert a == {"x": {"y": 1, "z": 2}}


def test_aux_configs_label_every_cost_as_assumption() -> None:
    costs = load_aux("costs")
    for section in ("items", "energy", "policy"):
        for key, item in costs[section].items():
            assert "assumption" in item["note"], key
    assert load_aux("acceptance")["status"] == "illustrative"


def test_seed_derivation_is_stable_and_keyed() -> None:
    assert derive_seed(1, "unit", 3) == derive_seed(1, "unit", 3)
    assert derive_seed(1, "unit", 3) != derive_seed(1, "unit", 4)
    a = make_rng(7, "x").normal(size=5)
    b = make_rng(7, "x").normal(size=5)
    np.testing.assert_array_equal(a, b)


def test_hashes_are_canonical() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
    assert len(sha256_bytes(b"x")) == 64


def test_run_folder_records_provenance(tiny_cfg) -> None:
    run = create_run(tiny_cfg)
    prov = json.loads((run.path / "provenance.json").read_text())
    assert {"git", "data_manifest_sha256", "torch"} <= prov.keys()
    assert run.config["name"] == "tiny"
    run.write_metrics("stage_a", {"f1": 0.5})
    run.write_metrics("stage_b", {"auc": np.float32(0.9)})
    assert set(run.read_metrics()) == {"stage_a", "stage_b"}
    assert latest_run(tiny_cfg).path == run.path
