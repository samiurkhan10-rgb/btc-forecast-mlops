"""Unit tests for the Flask API endpoints (Flask test client, no network, no gunicorn)."""

import json
import shutil

import pandas as pd
import pytest

from src.features import MIN_HISTORY
from src.registry import promote
from src.train_forecast import train


# ---------- no model on disk yet ----------
def test_health_without_model_is_still_200(client_no_model):
    resp = client_no_model.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["model_version"] is None
    assert body["uptime_seconds"] >= 0


def test_predict_without_model_returns_503(client_no_model):
    resp = client_no_model.post("/predict", json={})
    assert resp.status_code == 503
    assert "error" in resp.get_json()


def test_metrics_without_model_returns_503(client_no_model):
    assert client_no_model.get("/metrics").status_code == 503


# ---------- with a trained model ----------
def test_health_reports_version(client, trained):
    body = client.get("/health").get_json()
    assert body["model_loaded"] is True
    assert body["model_version"] == trained["version"]


def test_predict_default_seven_days(client):
    resp = client.post("/predict")  # empty body -> defaults
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["forecast_days"] == 7
    assert body["input_source"] == "model_history"
    preds = body["predictions"]
    assert len(preds) == 7
    dates = pd.to_datetime([p["date"] for p in preds])
    assert (dates.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert dates[0] == pd.Timestamp(body["last_observed_date"]) + pd.Timedelta(days=1)
    assert all(p["predicted_close"] > 0 for p in preds)


@pytest.mark.parametrize("days", [1, 14, 30])
def test_predict_custom_horizon(client, days):
    resp = client.post("/predict", json={"forecast_days": days})
    assert resp.status_code == 200
    assert len(resp.get_json()["predictions"]) == days


def test_predict_with_custom_history(client):
    history = [50_000 + 100 * i for i in range(MIN_HISTORY)]
    resp = client.post("/predict", json={"history": history, "forecast_days": 3})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["input_source"] == "custom_history"
    assert body["last_observed_close"] == history[-1]
    assert len(body["predictions"]) == 3


@pytest.mark.parametrize(
    "payload, fragment",
    [
        ({"forecast_days": 0}, "between 1 and 30"),
        ({"forecast_days": 31}, "between 1 and 30"),
        ({"forecast_days": "7"}, "integer"),
        ({"forecast_days": 7.5}, "integer"),
        ({"forecast_days": True}, "integer"),
        ({"forecast_days": None}, "integer"),
        ({"history": [1.0, 2.0]}, "between"),
        ({"history": "abc"}, "array"),
        ({"history": [50_000.0] * (MIN_HISTORY - 1) + ["x"]}, "finite positive"),
        ({"history": [50_000.0] * (MIN_HISTORY - 1) + [-5]}, "finite positive"),
        ({"unexpected": 1}, "unknown field"),
        ([1, 2, 3], "JSON object"),
    ],
)
def test_predict_validation_errors(client, payload, fragment):
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert fragment in resp.get_json()["error"]


def test_predict_invalid_json(client):
    resp = client.post("/predict", data="{not json", content_type="application/json")
    assert resp.status_code == 400
    assert "valid JSON" in resp.get_json()["error"]


def test_metrics_endpoint(client, trained):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["model_version"] == trained["version"]
    for key in ("mae", "rmse", "mape", "holdout_start", "holdout_end", "baseline_naive"):
        assert key in body["metrics"]


# ---------- reload ----------
def test_reload_requires_api_key(client):
    assert client.post("/reload").status_code == 401
    assert client.post("/reload", headers={"X-API-Key": "wrong"}).status_code == 401


def test_reload_disabled_without_configured_key(client, monkeypatch):
    monkeypatch.delenv("API_KEY")
    assert client.post("/reload", headers={"X-API-Key": "x"}).status_code == 403


def test_reload_picks_up_new_version(client, trained, features_csv, api_key):
    new = train(features_csv, trained["models_root"] / "staging")
    promote(new["model_dir"], trained["models_root"], "production")
    resp = client.post("/reload", headers={"X-API-Key": api_key})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["previous_version"] == trained["version"]
    assert body["model_version"] == new["version"]
    assert body["changed"] is True
    assert client.get("/health").get_json()["model_version"] == new["version"]


def test_other_workers_detect_new_version_without_reload(client, trained, features_csv):
    """Simulates a second gunicorn worker that never received POST /reload."""
    store = client.application.extensions["model_store"]
    new = train(features_csv, trained["models_root"] / "staging")
    promote(new["model_dir"], trained["models_root"], "production")
    store.refresh_if_changed(force=True)
    assert store.model.version == new["version"]


def test_reload_when_model_disappears_keeps_serving(client, trained, api_key):
    shutil.rmtree(trained["models_root"] / "production")
    resp = client.post("/reload", headers={"X-API-Key": api_key})
    assert resp.status_code == 200  # old model is kept in memory
    assert client.post("/predict").status_code == 200


def test_corrupt_model_is_rejected_and_old_one_kept(client, trained, api_key):
    prod = trained["models_root"] / "production"
    (prod / "feature_meta.json").write_text(json.dumps({"feature_columns": ["other"]}))
    (prod / "version.txt").write_text("broken-version")
    client.post("/reload", headers={"X-API-Key": api_key})
    assert client.get("/health").get_json()["model_version"] == trained["version"]


# ---------- generic error handling ----------
def test_unknown_route_returns_json_404(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert resp.get_json()["status"] == 404


def test_wrong_method_returns_json_405(client):
    assert client.get("/predict").status_code == 405


def test_internal_error_returns_json_500(client, monkeypatch):
    import app.app as api

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(api, "recursive_forecast", boom)
    resp = client.post("/predict", json={})
    assert resp.status_code == 500
    assert resp.get_json() == {"error": "internal server error", "status": 500}
