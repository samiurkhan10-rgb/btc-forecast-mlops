"""Recursive multi-step forecasting, shared by training evaluation and the Flask API."""

import math

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import MIN_HISTORY, next_step_vector


def recursive_forecast(
    booster: xgb.Booster, closes, last_date, horizon: int
) -> list[tuple[pd.Timestamp, float]]:
    """Predict `horizon` days ahead, one day at a time.

    Each step predicts tomorrow's log return, converts it to a price, appends that
    price to the history and recomputes the features for the following day.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    history = [float(c) for c in np.asarray(closes, dtype=float)[-MIN_HISTORY:]]
    last_date = pd.Timestamp(last_date).normalize()
    out = []
    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)
        # NumPy row in FEATURE_COLUMNS order (no per-step DataFrame: see next_step_vector)
        x = next_step_vector(history, next_date)
        log_ret = float(booster.inplace_predict(x)[0])
        price = history[-1] * math.exp(log_ret)
        history = history[1:] + [price]
        out.append((next_date, price))
    return out
