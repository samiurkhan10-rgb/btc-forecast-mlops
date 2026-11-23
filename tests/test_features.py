import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    MIN_HISTORY,
    make_training_frame,
    next_step_features,
)


def test_training_frame_shape_and_no_nans(raw_df):
    frame = make_training_frame(raw_df)
    assert set(FEATURE_COLUMNS) <= set(frame.columns)
    assert {"ma_7", "ma_30", "vol_7", "target"} <= set(frame.columns)
    assert not frame[FEATURE_COLUMNS].isna().any().any()
    # first complete row needs MIN_HISTORY closes; last row keeps target NaN (tomorrow unknown)
    assert len(frame) == len(raw_df) - (MIN_HISTORY - 1)
    assert frame["target"].isna().sum() == 1
    assert pd.isna(frame["target"].iloc[-1])


def test_moving_average_values(raw_df):
    frame = make_training_frame(raw_df)
    row = frame.iloc[0]
    idx = raw_df.index[raw_df["date"] == row["date"]][0]
    expected_ma7 = raw_df["close"].iloc[idx - 6 : idx + 1].mean()
    assert row["ma_7"] == pytest.approx(expected_ma7)


def test_target_is_next_day_log_return(raw_df):
    frame = make_training_frame(raw_df)
    c = raw_df.set_index("date")["close"]
    d = frame["date"].iloc[10]
    assert frame["target"].iloc[10] == pytest.approx(np.log(c[d + pd.Timedelta(days=1)] / c[d]))


def test_no_lookahead_leakage(raw_df):
    """Changing FUTURE prices must not change today's features."""
    base = make_training_frame(raw_df)
    altered = raw_df.copy()
    altered.loc[altered.index[-50:], "close"] *= 3
    changed = make_training_frame(altered)
    n = len(base) - 50
    pd.testing.assert_frame_equal(base[FEATURE_COLUMNS].iloc[:n], changed[FEATURE_COLUMNS].iloc[:n])


def test_inference_features_match_training_features(raw_df):
    """Train/serve parity: next_step_features == the training row for the same day."""
    frame = make_training_frame(raw_df)
    i = 100
    day = frame["date"].iloc[i]
    closes = raw_df.loc[raw_df["date"] <= day, "close"].to_numpy()
    x = next_step_features(closes, day + pd.Timedelta(days=1))
    np.testing.assert_allclose(
        x.iloc[0].to_numpy(dtype=float), frame[FEATURE_COLUMNS].iloc[i].to_numpy(dtype=float)
    )


def test_next_step_features_needs_min_history():
    with pytest.raises(ValueError):
        next_step_features([1.0] * (MIN_HISTORY - 1), pd.Timestamp("2026-01-01"))


def test_numpy_fast_path_matches_pandas_on_every_row(raw_df):
    """next_step_vector (NumPy, used by the API) == indicator_frame (pandas, used in training)."""
    from src.features import next_step_vector

    frame = make_training_frame(raw_df)
    closes = raw_df["close"].to_numpy()
    for i in range(0, len(frame), 7):
        day = frame["date"].iloc[i]
        upto = closes[: raw_df.index[raw_df["date"] == day][0] + 1]
        vec = next_step_vector(upto, day + pd.Timedelta(days=1))
        np.testing.assert_allclose(
            vec[0], frame[FEATURE_COLUMNS].iloc[i].to_numpy(dtype=float), rtol=1e-9, atol=1e-12
        )
