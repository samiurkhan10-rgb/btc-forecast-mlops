"""Flask REST API serving the 7-day BTC close-price forecast.

Endpoints
  GET  /health   liveness/readiness: status, uptime, model_version, model_loaded (always 200)
  POST /predict  {"forecast_days": 1-30 (default 7), "history": [floats] (optional)}
  GET  /metrics  MAE / RMSE / MAPE of the latest validation run
  POST /reload   re-read model files from disk (header X-API-Key must match API_KEY)

Production server:  gunicorn --bind 0.0.0.0:5000 --workers 2 app.app:app
"""

import hmac
import logging
import math
import os
import time
from pathlib import Path

import pandas as pd
from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from app.model_store import ModelStore
from src.features import MIN_HISTORY
from src.forecast import recursive_forecast
from src.logging_config import setup_logging

log = logging.getLogger("btc_api")

DEFAULT_FORECAST_DAYS = 7
MAX_FORECAST_DAYS = 30
MAX_HISTORY = 5000


class ValidationError(ValueError):
    """Bad client input -> HTTP 400."""


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_predict_payload(payload) -> tuple[int, list | None]:
    """Validate the /predict body. Returns (forecast_days, history or None)."""
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValidationError("request body must be a JSON object")
    unknown = set(payload) - {"forecast_days", "history"}
    if unknown:
        raise ValidationError(f"unknown field(s): {sorted(unknown)}")

    days = payload.get("forecast_days", DEFAULT_FORECAST_DAYS)
    if days is None or not isinstance(days, int) or isinstance(days, bool):
        raise ValidationError("forecast_days must be an integer")
    if not 1 <= days <= MAX_FORECAST_DAYS:
        raise ValidationError(f"forecast_days must be between 1 and {MAX_FORECAST_DAYS}")

    history = payload.get("history")
    if history is not None:
        if not isinstance(history, list):
            raise ValidationError("history must be an array of numbers")
        if not MIN_HISTORY <= len(history) <= MAX_HISTORY:
            raise ValidationError(
                f"history must contain between {MIN_HISTORY} and {MAX_HISTORY} prices"
            )
        if not all(_is_number(v) and math.isfinite(v) and v > 0 for v in history):
            raise ValidationError("history values must be finite positive numbers")
        history = [float(v) for v in history]
    return days, history


def create_app(models_dir: str | Path | None = None, stage: str | None = None) -> Flask:
    setup_logging()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024  # reject huge bodies with 413
    app.json.sort_keys = False

    store = ModelStore(
        Path(models_dir or os.getenv("MODELS_DIR", "/app/models")),
        stage or os.getenv("MODEL_STAGE", "production"),
    )
    store.load()
    started = time.monotonic()
    app.extensions["model_store"] = store

    def error(status: int, message: str, **extra):
        return jsonify({"error": message, "status": status, **extra}), status

    @app.before_request
    def _before():
        g.t0 = time.perf_counter()
        try:
            store.refresh_if_changed()
        except Exception:  # noqa: BLE001 - never fail a request because of the check
            log.exception("model refresh check failed")

    @app.after_request
    def _after(resp):
        fields = {
            "method": request.method,
            "path": request.path,
            "status": resp.status_code,
            "duration_ms": round(
                (time.perf_counter() - g.get("t0", time.perf_counter())) * 1000, 2
            ),
        }
        # probes hit /health every few seconds: keep them out of INFO logs
        log.log(
            logging.DEBUG if request.path == "/health" else logging.INFO, "request", extra=fields
        )
        return resp

    @app.get("/health")
    def health():
        m = store.model
        return jsonify(
            {
                "status": "ok",
                "uptime_seconds": round(time.monotonic() - started, 1),
                "model_loaded": m is not None,
                "model_version": m.version if m else None,
                "model_stage": store.stage_dir.name,
            }
        )

    @app.post("/predict")
    def predict():
        raw = request.get_data(cache=True)
        payload = request.get_json(force=True, silent=True) if raw.strip() else {}
        if payload is None:
            return error(400, "request body is not valid JSON")
        try:
            days, history = parse_predict_payload(payload)
        except ValidationError as exc:
            return error(400, str(exc))

        m = store.model
        if m is None:
            return error(503, "model not loaded yet - run the training pipeline first")

        if history is None:
            closes, last_date, source = m.closes, m.last_date, "model_history"
        else:
            # custom history: assume the last value is yesterday's (latest complete) close
            closes = history
            today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
            last_date = today - pd.Timedelta(days=1)
            source = "custom_history"

        preds = recursive_forecast(m.booster, closes, last_date, days)
        return jsonify(
            {
                "model_version": m.version,
                "forecast_days": days,
                "input_source": source,
                "last_observed_date": str(last_date.date()),
                "last_observed_close": round(float(closes[-1]), 2),
                "predictions": [
                    {"date": str(d.date()), "predicted_close": round(p, 2)} for d, p in preds
                ],
            }
        )

    @app.get("/metrics")
    def metrics():
        m = store.model
        if m is None:
            return error(503, "model not loaded yet - no metrics available")
        return jsonify({"model_version": m.version, "metrics": m.metrics})

    @app.post("/reload")
    def reload_model():
        expected = os.getenv("API_KEY", "")
        if not expected:
            return error(403, "reload disabled: API_KEY is not configured")
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided.encode(), expected.encode()):
            return error(401, "invalid or missing X-API-Key")
        previous = store.model.version if store.model else None
        store.load()
        m = store.model
        if m is None:
            return error(503, "no model available on disk", model_loaded=False)
        return jsonify(
            {
                "reloaded": True,
                "previous_version": previous,
                "model_version": m.version,
                "changed": previous != m.version,
            }
        )

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return error(exc.code or 500, exc.description or exc.name)

    @app.errorhandler(Exception)
    def unhandled(exc: Exception):
        log.exception("unhandled error", extra={"path": request.path})
        return error(500, "internal server error")

    return app


app = create_app()
