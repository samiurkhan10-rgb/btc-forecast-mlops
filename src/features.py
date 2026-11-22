"""Transform step: technical indicators + model features.

Design note (important for tree models): XGBoost cannot extrapolate beyond the
target values it saw in training, and BTC prices trend to new highs. So the model
predicts the next day's *log return* from scale-free features (lagged returns,
price/moving-average ratios, volatility). The price is rebuilt as
close_t * exp(predicted_return).

The SAME function (`indicator_frame`) builds features for training and for
inference, which prevents training/serving skew.
"""

from pathlib import Path

import numpy as np
import pandas as pd

N_RETURN_LAGS = 7
MA_SHORT = 7
MA_LONG = 30
# closes needed to compute one complete feature row (30-day return volatility needs 31 closes)
MIN_HISTORY = MA_LONG + 1

FEATURE_COLUMNS = [f"ret_lag_{i}" for i in range(1, N_RETURN_LAGS + 1)] + [
    "ma7_ratio",
    "ma30_ratio",
    "vol_7",
    "vol_30",
    "dow",
]


def indicator_frame(close: pd.Series) -> pd.DataFrame:
    """Indicators computed at day t using only closes up to and including day t."""
    close = close.astype(float).reset_index(drop=True)
    log_ret = np.log(close).diff()
    out = pd.DataFrame(index=close.index)
    for i in range(1, N_RETURN_LAGS + 1):
        out[f"ret_lag_{i}"] = log_ret.shift(i - 1)  # ret_lag_1 = today's return
    out["ma_7"] = close.rolling(MA_SHORT).mean()
    out["ma_30"] = close.rolling(MA_LONG).mean()
    out["ma7_ratio"] = close / out["ma_7"] - 1.0
    out["ma30_ratio"] = close / out["ma_30"] - 1.0
    out["vol_7"] = log_ret.rolling(MA_SHORT).std()
    out["vol_30"] = log_ret.rolling(MA_LONG).std()
    return out


def make_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw OHLCV -> one row per day t with features and target = log return of day t+1."""
    df = raw[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    feats = indicator_frame(df["close"])
    feats.insert(0, "date", df["date"])
    feats.insert(1, "close", df["close"].astype(float))
    target_date = df["date"] + pd.Timedelta(days=1)
    feats["dow"] = target_date.dt.dayofweek  # weekday of the day being predicted
    feats["target"] = np.log(df["close"].astype(float)).diff().shift(-1)
    # Rows without a full look-back window are dropped. The LAST row keeps target=NaN
    # (tomorrow is unknown) so the file still contains the latest close; training
    # filters target.notna() itself.
    return feats.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def next_step_features(closes, next_date: pd.Timestamp) -> pd.DataFrame:
    """One feature row for predicting `next_date` given the closes up to the day before."""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} closes, got {len(closes)}")
    feats = indicator_frame(pd.Series(closes[-MIN_HISTORY:])).iloc[[-1]].copy()
    feats["dow"] = pd.Timestamp(next_date).dayofweek
    return feats[FEATURE_COLUMNS].reset_index(drop=True)


def run_features(raw_path: Path, out_path: Path) -> Path:
    """Entry point for Airflow / CLI / KFP: raw CSV -> processed features CSV."""
    from src.etl_fetch import save_csv_atomic  # local import keeps API deps minimal

    raw = pd.read_csv(raw_path, parse_dates=["date"])
    frame = make_training_frame(raw)
    if frame.empty:
        raise ValueError("feature frame is empty - not enough raw data")
    return save_csv_atomic(frame, out_path)
