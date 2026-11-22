"""Train the 7-day BTC close forecaster (XGBoost on next-day log returns).

1. Hold out the last 7 days, train on everything before, forecast those 7 days
   RECURSIVELY (exactly like the API does) -> honest multi-step MAE / RMSE / MAPE.
   A naive "tomorrow = today" baseline is computed on the same window for context.
2. Re-fit on all data and save a versioned artifact directory:
   model.json, metrics.json, history.json, feature_meta.json, version.txt
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config
from src.features import FEATURE_COLUMNS, MIN_HISTORY
from src.forecast import recursive_forecast

log = logging.getLogger(__name__)

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": 2,  # the laptop has 2 cores
    "seed": 42,
}
NUM_BOOST_ROUND = 300
HISTORY_TO_STORE = 90  # closes saved with the model so /predict works with no input


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err / y_true)) * 100.0),
    }


def fit_booster(frame: pd.DataFrame) -> xgb.Booster:
    train = frame[frame["target"].notna()]
    if len(train) < 50:
        raise ValueError(f"not enough training rows: {len(train)}")
    dtrain = xgb.DMatrix(train[FEATURE_COLUMNS], label=train["target"])
    return xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)


def evaluate_holdout(frame: pd.DataFrame, horizon: int = config.HOLDOUT_DAYS) -> dict:
    """Train on data before the last `horizon` days and forecast them recursively."""
    frame = frame.sort_values("date").reset_index(drop=True)
    cut = len(frame) - horizon  # first holdout row
    if cut < MIN_HISTORY + 50:
        raise ValueError("not enough data for a holdout evaluation")
    # Row i's target is close[i+1]; rows up to cut-2 only use targets before the holdout.
    booster = fit_booster(frame.iloc[: cut - 1])
    preds = recursive_forecast(
        booster, frame["close"].iloc[:cut], frame["date"].iloc[cut - 1], horizon
    )
    actual = frame["close"].iloc[cut:].to_numpy()
    predicted = np.array([p for _, p in preds])
    naive = np.full(horizon, frame["close"].iloc[cut - 1])
    return {
        **compute_metrics(actual, predicted),
        "baseline_naive": compute_metrics(actual, naive),
        "holdout_start": str(frame["date"].iloc[cut].date()),
        "holdout_end": str(frame["date"].iloc[-1].date()),
        "holdout_days": horizon,
        "holdout_actual": [round(float(v), 2) for v in actual],
        "holdout_predicted": [round(float(v), 2) for v in predicted],
    }


def _new_version(root: Path) -> str:
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version, n = base, 0
    while (root / version).exists():
        n += 1
        version = f"{base}-{n}"
    return version


def train(features_csv: Path, out_root: Path | None = None) -> dict:
    """Entry point for Airflow / CLI / KFP. Returns {version, model_dir, metrics}."""
    frame = pd.read_csv(features_csv, parse_dates=["date"]).sort_values("date")
    frame = frame.reset_index(drop=True)
    metrics = evaluate_holdout(frame)

    booster = fit_booster(frame)  # final model on all available data
    out_root = Path(out_root or config.models_dir() / "staging")
    version = _new_version(out_root)
    model_dir = out_root / version
    model_dir.mkdir(parents=True)

    metrics.update(
        {
            "model_version": version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_type": "xgboost (next-day log return, recursive multi-step)",
            "n_train_rows": int(frame["target"].notna().sum()),
            "data_start": str(frame["date"].iloc[0].date()),
            "data_end": str(frame["date"].iloc[-1].date()),
        }
    )
    tail = frame.tail(HISTORY_TO_STORE)
    history = {
        "dates": [str(d.date()) for d in tail["date"]],
        "closes": [float(c) for c in tail["close"]],
    }
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "min_history": MIN_HISTORY,
        "xgb_params": XGB_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
    }
    booster.save_model(str(model_dir / config.MODEL_FILE))
    (model_dir / config.METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    (model_dir / config.HISTORY_FILE).write_text(json.dumps(history))
    (model_dir / config.META_FILE).write_text(json.dumps(meta, indent=2))
    # version.txt is written LAST: its presence marks a complete artifact directory
    (model_dir / config.VERSION_FILE).write_text(version)

    log.info(
        "model trained",
        extra={
            "version": version,
            "mae": round(metrics["mae"], 2),
            "rmse": round(metrics["rmse"], 2),
            "mape": round(metrics["mape"], 3),
            "baseline_mape": round(metrics["baseline_naive"]["mape"], 3),
        },
    )
    return {"version": version, "model_dir": str(model_dir), "metrics": metrics}
