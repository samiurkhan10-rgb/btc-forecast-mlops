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


def next_step_vector(closes, next_date) -> np.ndarray:
    """Fast inference path: ONE feature row as a (1, n_features) NumPy array.

    Computes exactly what `indicator_frame` computes for its last row, but with plain NumPy
    on the last MIN_HISTORY closes. Building a pandas DataFrame per step cost ~47 ms vs
    ~0.1 ms here, which dominated /predict latency under load (found in the Part D load test).
    Equality with the pandas version is enforced by tests/test_features.py.
    """
    c = np.asarray(closes, dtype=float)
    if len(c) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} closes, got {len(c)}")
    c = c[-MIN_HISTORY:]
    r = np.diff(np.log(c))  # MIN_HISTORY - 1 = 30 daily log returns
    lags = r[::-1][:N_RETURN_LAGS]  # ret_lag_1 = latest return, ret_lag_2 = the one before...
    row = [
        *lags,
        c[-1] / c[-MA_SHORT:].mean() - 1.0,  # ma7_ratio
        c[-1] / c[-MA_LONG:].mean() - 1.0,  # ma30_ratio
        r[-MA_SHORT:].std(ddof=1),  # vol_7 (pandas rolling std uses ddof=1)
        r[-MA_LONG:].std(ddof=1),  # vol_30
        pd.Timestamp(next_date).dayofweek,  # dow of the predicted day
    ]
    return np.asarray(row, dtype=float).reshape(1, -1)


def next_step_features(closes, next_date: pd.Timestamp) -> pd.DataFrame:
    """Same row as `next_step_vector`, labelled with FEATURE_COLUMNS (for inspection/tests)."""
    return pd.DataFrame(next_step_vector(closes, next_date), columns=FEATURE_COLUMNS)


def run_features(raw_path: Path, out_path: Path) -> Path:
    """Entry point for Airflow / CLI / KFP: raw CSV -> processed features CSV."""
    from src.etl_fetch import save_csv_atomic  # local import keeps API deps minimal

    raw = pd.read_csv(raw_path, parse_dates=["date"])
    frame = make_training_frame(raw)
    if frame.empty:
        raise ValueError("feature frame is empty - not enough raw data")
    return save_csv_atomic(frame, out_path)
