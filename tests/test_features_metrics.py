"""Flow-curve fitting, metrics and the tabular pipeline on the tiny profile."""

from __future__ import annotations

import numpy as np
import pandas as pd

from preinspect.evaluate.metrics import expected_calibration_error, outcome_metrics
from preinspect.features.flow import LoadResidual, fit_curve
from preinspect.models.tabular import PROBA_COLUMNS
from preinspect.runs import latest_run


def test_fit_curve_recovers_coefficients_with_dropouts() -> None:
    q = np.linspace(0.03, 0.24, 8)
    dp = 7.0 * q + 40.0 * q**2
    dp[[1, 5]] = np.nan
    a, b, rms, n = fit_curve(q, dp)
    assert abs(a - 7.0) < 1e-6 and abs(b - 40.0) < 1e-6 and n == 6 and rms < 1e-9
    assert np.isnan(fit_curve(q, np.full(8, np.nan))[0])


def test_load_residual_flags_low_restriction() -> None:
    load = pd.Series(np.linspace(5, 40, 50))
    ratio = 1.0 + 0.05 * load
    model = LoadResidual().fit(load, ratio)
    leaking = model.transform(pd.Series([20.0]), pd.Series([1.2]))
    assert leaking.iloc[0] < -0.5


def test_metrics_on_perfect_and_uninformative_predictions() -> None:
    y = np.array(["PASS", "REWORK", "SCRAP"] * 10)
    onehot = np.eye(3)[np.tile([0, 1, 2], 10)]
    m = outcome_metrics(y, onehot)
    assert m["macro_f1"] == 1.0 and m["auc_macro"] == 1.0
    assert expected_calibration_error(y, onehot) == 0.0
    flat = np.full((30, 3), 1 / 3)
    assert abs(expected_calibration_error(y, flat) - 0.0) < 1e-9


def test_train_and_evaluate_tiny(tiny_data) -> None:
    from preinspect.stages import evaluate, train

    run = train.run(tiny_data)
    evaluate.run(tiny_data, run.name)
    metrics = latest_run(tiny_data).read_metrics()
    models = metrics["evaluate"]["models"]
    assert {"metadata_only", "classical_flow_meta"} <= models.keys()
    assert 0.0 <= models["classical_flow_meta"]["truth"]["macro_f1"] <= 1.0
    preds = pd.read_parquet(run.path / "predictions" / "flow_only.parquet")
    assert np.allclose(preds[PROBA_COLUMNS].sum(axis=1), 1.0)
