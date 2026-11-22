"""ETL tests: yfinance, CoinGecko and Binance are always mocked (CI never hits external APIs)."""

import sys
import types
from datetime import datetime

import pandas as pd
import pytest

from src import etl_fetch
from tests.conftest import make_prices


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Keep retry logic but make the backoff instant in tests."""
    sleeps = []
    monkeypatch.setattr(etl_fetch.time, "sleep", sleeps.append)
    return sleeps


def _binance_klines(n=120):
    start = pd.Timestamp("2026-06-01", tz="UTC")
    rows = []
    for i in range(n):
        t = int((start + pd.Timedelta(days=i)).timestamp() * 1000)
        o, c = 60_000 + 10 * i, 60_005 + 10 * i
        rows.append([t, str(o), str(c + 50), str(o - 50), str(c), "12.5", t + 86_399_999, "7.5e8"])
    return rows


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


def test_fetch_binance_parses_klines(monkeypatch):
    monkeypatch.setattr(
        etl_fetch.requests, "get", lambda *a, **k: FakeResponse(_binance_klines(100))
    )
    df = etl_fetch.fetch_binance()
    assert list(df.columns) == etl_fetch.COLUMNS
    assert len(df) == 100
    first = df.iloc[0]
    assert float(first["high"]) >= float(first["close"]) >= float(first["low"])
    assert float(first["volume"]) == pytest.approx(7.5e8)  # quote (USD) volume


def test_with_retries_recovers_after_transient_failures(_no_retry_sleep):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("slow network")
        return "ok"

    assert etl_fetch.with_retries("x", flaky, attempts=3, backoff=5) == "ok"
    assert _no_retry_sleep == [5, 15]  # exponential backoff 5 s, then 15 s


def test_with_retries_gives_up_after_last_attempt():
    def always_fail():
        raise TimeoutError("down")

    with pytest.raises(TimeoutError):
        etl_fetch.with_retries("x", always_fail, attempts=2, backoff=0)


def test_yfinance_is_retried_before_falling_back(monkeypatch):
    """An empty yfinance result (its symptom of a network timeout) is retried."""
    good = _yf_style(make_prices(120))
    results = [pd.DataFrame(), good]
    fake = types.ModuleType("yfinance")

    class Ticker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            assert kwargs["timeout"] == etl_fetch.YF_TIMEOUT
            return results.pop(0)

    fake.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    df, source = etl_fetch.fetch_btc()
    assert source == "yfinance"
    assert len(df) == 120


def test_falls_back_to_binance_when_coingecko_fails(monkeypatch):
    monkeypatch.setattr(etl_fetch, "fetch_yfinance", lambda: pd.DataFrame())

    def fake_get(url, **kwargs):
        if "coingecko" in url:
            return FakeResponse({}, status=503)
        return FakeResponse(_binance_klines())

    monkeypatch.setattr(etl_fetch.requests, "get", fake_get)
    df, source = etl_fetch.fetch_btc()
    assert source == "binance"
    assert len(df) == 120
