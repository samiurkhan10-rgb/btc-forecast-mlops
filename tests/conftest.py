"""Shared fixtures. All tests run OFFLINE: real network connections are blocked."""

import socket

import numpy as np
import pandas as pd
import pytest

from src.features import run_features
from src.registry import promote
from src.train_forecast import train


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Any accidental real network call fails loudly instead of hitting yfinance/CoinGecko."""

    def guard(*args, **kwargs):
        raise RuntimeError("network access is disabled in tests - mock it")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


def make_prices(n_days: int = 400, seed: int = 7, end: str = "2026-11-01") -> pd.DataFrame:
    """Synthetic geometric random walk that looks like daily BTC OHLCV."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=n_days, freq="D")
    close = 60_000 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n_days)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.uniform(1e9, 5e9, n_days),
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def features_csv(tmp_path, raw_df):
    raw_path = tmp_path / "data" / "raw.csv"
    raw_path.parent.mkdir(parents=True)
    raw_df.to_csv(raw_path, index=False)
    return run_features(raw_path, tmp_path / "data" / "features.csv")


@pytest.fixture
def trained(tmp_path, features_csv):
    """Train once on synthetic data and promote to <tmp>/models/production."""
    models_root = tmp_path / "models"
    result = train(features_csv, models_root / "staging")
    promote(result["model_dir"], models_root, "production")
    return {"models_root": models_root, **result}


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    return "test-key"


@pytest.fixture
def client_no_model(tmp_path, api_key):
    from app.app import create_app

    app = create_app(models_dir=tmp_path / "empty_models", stage="production")
    return app.test_client()


@pytest.fixture
def client(trained, api_key):
    from app.app import create_app

    app = create_app(models_dir=trained["models_root"], stage="production")
    return app.test_client()
