import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src import config
from src.forecast import recursive_forecast
from src.train_forecast import compute_metrics, evaluate_holdout, train


def test_compute_metrics_known_values():
    m = compute_metrics([100, 200], [110, 180])
    assert m["mae"] == pytest.approx(15.0)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["mape"] == pytest.approx((10 + 10) / 2)


def test_holdout_uses_last_seven_days(features_csv):
    frame = pd.read_csv(features_csv, parse_dates=["date"])
    m = evaluate_holdout(frame)
    assert m["holdout_days"] == 7
    assert m["holdout_end"] == str(frame["date"].iloc[-1].date())
    assert len(m["holdout_predicted"]) == 7
    assert m["mape"] >= 0 and np.isfinite(m["mape"])


def test_train_writes_complete_artifact_dir(trained):
    d = trained["models_root"] / "staging" / trained["version"]
    for name in (
        config.MODEL_FILE,
        config.METRICS_FILE,
        config.HISTORY_FILE,
        config.META_FILE,
        config.VERSION_FILE,
    ):
        assert (d / name).is_file(), name
    assert (d / config.VERSION_FILE).read_text() == trained["version"]
    metrics = json.loads((d / config.METRICS_FILE).read_text())
    assert metrics["model_version"] == trained["version"]
    assert {"mae", "rmse", "mape", "baseline_naive"} <= set(metrics)


def test_two_trainings_get_distinct_versions(tmp_path, features_csv):
    a = train(features_csv, tmp_path / "staging")
    b = train(features_csv, tmp_path / "staging")
    assert a["version"] != b["version"]


def test_recursive_forecast_dates_and_values(trained):
    booster = xgb.Booster()
    d = trained["models_root"] / "production"
    booster.load_model(str(d / config.MODEL_FILE))
    hist = json.loads((d / config.HISTORY_FILE).read_text())
    out = recursive_forecast(booster, hist["closes"], hist["dates"][-1], 7)
    assert len(out) == 7
    assert out[0][0] == pd.Timestamp(hist["dates"][-1]) + pd.Timedelta(days=1)
    # one-step-ahead log returns are small, so prices stay within a sane band
    assert all(0.5 * hist["closes"][-1] < p < 1.5 * hist["closes"][-1] for _, p in out)


def test_recursive_forecast_rejects_bad_horizon(trained):
    booster = xgb.Booster()
    booster.load_model(str(trained["models_root"] / "production" / config.MODEL_FILE))
    with pytest.raises(ValueError):
        recursive_forecast(booster, [1.0] * 40, "2026-01-01", 0)
