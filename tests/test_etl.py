"""ETL tests: yfinance and CoinGecko are always mocked (CI must never hit external APIs)."""

import sys
import types
from datetime import datetime

import pandas as pd
import pytest

from src import etl_fetch
from tests.conftest import make_prices


def _fake_yfinance(monkeypatch, history_df):
    """Install a fake `yfinance` module whose Ticker(...).history() returns history_df."""
    fake = types.ModuleType("yfinance")

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return history_df

    fake.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def _yf_style(df: pd.DataFrame) -> pd.DataFrame:
    """Shape like yfinance: tz-aware DatetimeIndex 'Date', capitalised columns, extras."""
    out = df.set_index(pd.DatetimeIndex(df["date"]).tz_localize("UTC").rename("Date"))
    out = out.drop(columns="date").rename(columns=str.capitalize)
    out["Dividends"] = 0.0
    out["Stock Splits"] = 0.0
    return out


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _coingecko_payload(n=120):
    start = pd.Timestamp("2026-06-01", tz="UTC")
    ts = [int((start + pd.Timedelta(days=i)).timestamp() * 1000) for i in range(n)]
    return {
        "prices": [[t, 60_000 + 10 * i] for i, t in enumerate(ts)],
        "total_volumes": [[t, 1e9] for t in ts],
    }


def test_fetch_yfinance_normalises_columns(monkeypatch):
    _fake_yfinance(monkeypatch, _yf_style(make_prices(100)))
    df = etl_fetch.fetch_yfinance()
    assert list(df.columns) == etl_fetch.COLUMNS
    assert len(df) == 100


def test_fetch_yfinance_empty_raises(monkeypatch):
    _fake_yfinance(monkeypatch, pd.DataFrame())
    with pytest.raises(ValueError):
        etl_fetch.fetch_yfinance()


def test_fetch_coingecko_builds_ohlcv(monkeypatch):
    calls = {}

    def fake_get(url, params, headers, timeout):
        calls["url"], calls["headers"] = url, headers
        return FakeResponse(_coingecko_payload())

    monkeypatch.setenv("COINGECKO_API_KEY", "demo-key")
    monkeypatch.setattr(etl_fetch.requests, "get", fake_get)
    df = etl_fetch.fetch_coingecko()
    assert list(df.columns) == etl_fetch.COLUMNS
    assert (df["high"] >= df["low"]).all()
    assert calls["headers"] == {"x-cg-demo-api-key": "demo-key"}


def test_clean_drops_today_duplicates_and_bad_rows():
    df = pd.DataFrame(
        {
            "date": [
                "2026-11-20",
                "2026-11-21",
                "2026-11-21",
                "2026-11-19",
                "2026-11-22",
                "2026-11-18",
            ],
            "open": [1, 1, 1, 1, 1, 1],
            "high": [1, 1, 1, 1, 1, 1],
            "low": [1, 1, 1, 1, 1, 1],
            "close": [100, 101, 102, -5, 103, 104],
            "volume": [1, 1, 1, 1, 1, 1],
        }
    )
    out = etl_fetch.clean(df, today=datetime(2026, 11, 22, 15, 0))
    assert out["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-11-18",
        "2026-11-20",
        "2026-11-21",
    ]
    assert out["close"].tolist() == [104, 100, 102]  # negative removed, dup keeps last


def test_fetch_btc_falls_back_to_coingecko(monkeypatch):
    def broken_yf():
        raise ConnectionError("yahoo down")

    monkeypatch.setattr(etl_fetch, "fetch_yfinance", broken_yf)
    monkeypatch.setattr(
        etl_fetch.requests, "get", lambda *a, **k: FakeResponse(_coingecko_payload())
    )
    df, source = etl_fetch.fetch_btc()
    assert source == "coingecko"
    assert len(df) >= etl_fetch.MIN_ROWS


def test_fetch_btc_all_sources_fail(monkeypatch):
    monkeypatch.setattr(etl_fetch, "fetch_yfinance", lambda: (_ for _ in ()).throw(ValueError("a")))
    monkeypatch.setattr(etl_fetch.requests, "get", lambda *a, **k: FakeResponse({}, status=429))
    with pytest.raises(RuntimeError, match="all data sources failed"):
        etl_fetch.fetch_btc()


def test_run_fetch_writes_csv(monkeypatch, tmp_path):
    _fake_yfinance(monkeypatch, _yf_style(make_prices(200)))
    path = etl_fetch.run_fetch(tmp_path / "raw" / "btc.csv")
    saved = pd.read_csv(path)
    assert list(saved.columns) == etl_fetch.COLUMNS
    assert len(saved) == 200
    assert not list((tmp_path / "raw").glob("*.tmp"))


def test_network_is_blocked_in_tests():
    import urllib.request

    with pytest.raises(Exception):
        urllib.request.urlopen("https://example.com", timeout=2)
