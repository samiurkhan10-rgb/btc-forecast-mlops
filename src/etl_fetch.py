"""Extract step: pull daily BTC-USD OHLCV data and store it in raw storage (CSV).

Sources are tried in order: yfinance -> CoinGecko -> Binance public API.
Each source gets several attempts with exponential backoff, because a slow or
flaky connection (multi-second TCP connects) otherwise fails a whole run.
The output always has the same schema:
    date, open, high, low, close, volume
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src import config

log = logging.getLogger(__name__)

COLUMNS = ["date", "open", "high", "low", "close", "volume"]
MIN_ROWS = 60  # refuse to store obviously broken downloads
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
BINANCE_URL = "https://api.binance.com/api/v3/klines"
YF_TIMEOUT = 60  # seconds; yfinance's default of 10 s is too short on slow links
HTTP_TIMEOUT = (20, 90)  # (connect, read) seconds for plain HTTP APIs


def _retries() -> int:
    return max(1, int(os.getenv("FETCH_RETRIES", "3")))


def _backoff() -> float:
    return float(os.getenv("FETCH_BACKOFF_SECONDS", "5"))


def fetch_yfinance(ticker: str = config.TICKER, period: str = "3y") -> pd.DataFrame:
    """Download daily candles with yfinance and normalise the columns."""
    import yfinance as yf  # imported lazily: not needed by the API image

    # yfinance swallows network errors and returns an empty frame (it logs
    # "possibly delisted"), so an empty result is treated as a retryable failure.
    hist = yf.Ticker(ticker).history(
        period=period, interval="1d", auto_adjust=False, timeout=YF_TIMEOUT
    )
    if hist is None or hist.empty:
        raise ValueError("yfinance returned no data (network timeout or rate limit)")
    df = hist.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"datetime": "date"})
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"yfinance data is missing columns: {sorted(missing)}")
    return df[COLUMNS]


def fetch_coingecko(days: int = 365, timeout=HTTP_TIMEOUT) -> pd.DataFrame:
    """Fallback source. The free market_chart endpoint gives daily close + volume only,
    so open is approximated by the previous close and high/low by max/min(open, close)."""
    headers = {}
    api_key = os.getenv("COINGECKO_API_KEY")
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    resp = requests.get(
        COINGECKO_URL,
        params={"vs_currency": "usd", "days": days, "interval": "daily"},
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    prices = pd.DataFrame(body.get("prices", []), columns=["ts", "close"])
    volumes = pd.DataFrame(body.get("total_volumes", []), columns=["ts", "volume"])
    if prices.empty:
        raise ValueError("CoinGecko returned no prices")
    df = prices.merge(volumes, on="ts", how="left")
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    return df[COLUMNS]


def fetch_binance(limit: int = 1000, timeout=HTTP_TIMEOUT) -> pd.DataFrame:
    """Last-resort source: Binance public klines (BTCUSDT ~ BTC-USD), true daily OHLCV.
    Volume uses the quote-asset (USDT) volume so it is in dollars like yfinance."""
    resp = requests.get(
        BINANCE_URL,
        params={"symbol": "BTCUSDT", "interval": "1d", "limit": limit},
        timeout=timeout,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise ValueError("Binance returned no klines")
    df = pd.DataFrame(
        [r[:8] for r in rows],
        columns=["ts", "open", "high", "low", "close", "base_volume", "close_ts", "volume"],
    )
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[COLUMNS]


def with_retries(name: str, fn, attempts: int | None = None, backoff: float | None = None):
    """Call fn() up to `attempts` times, sleeping backoff * 3**i between tries (5 s, 15 s...)."""
    attempts = attempts or _retries()
    backoff = _backoff() if backoff is None else backoff
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retry any network/parse failure
            if i == attempts - 1:
                raise
            wait = backoff * 3**i
            log.warning(
                "fetch attempt failed, retrying",
                extra={"source": name, "attempt": i + 1, "wait_s": wait, "error": str(exc)},
            )
            time.sleep(wait)


def clean(df: pd.DataFrame, today: datetime | None = None) -> pd.DataFrame:
    """Normalise dates to tz-naive calendar days, sort, de-duplicate, drop bad rows,
    and drop today's still-forming (incomplete) daily candle."""
    out = df.copy()
    dates = pd.to_datetime(out["date"], utc=True)
    out["date"] = dates.dt.tz_localize(None).dt.normalize()
    for col in COLUMNS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    out = out.sort_values("date").drop_duplicates(subset="date", keep="last")
    today = (today or datetime.now(timezone.utc)).replace(tzinfo=None)
    out = out[out["date"] < pd.Timestamp(today).normalize()]
    return out.reset_index(drop=True)


def fetch_btc() -> tuple[pd.DataFrame, str]:
    """Try yfinance, then CoinGecko, then Binance. Returns (cleaned dataframe, source name)."""
    errors = []
    sources = (
        ("yfinance", fetch_yfinance),
        ("coingecko", fetch_coingecko),
        ("binance", fetch_binance),
    )
    for name, fn in sources:
        try:
            df = clean(with_retries(name, fn))
            if len(df) < MIN_ROWS:
                raise ValueError(f"only {len(df)} rows")
            log.info("fetched BTC data", extra={"source": name, "rows": len(df)})
            return df, name
        except Exception as exc:  # noqa: BLE001 - any failure means "try the next source"
            log.warning("data source failed", extra={"source": name, "error": str(exc)})
            errors.append(f"{name}: {exc}")
    raise RuntimeError("all data sources failed: " + "; ".join(errors))


def save_csv_atomic(df: pd.DataFrame, path: Path) -> Path:
    """Write to a temp file then rename, so readers never see a half-written CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, date_format="%Y-%m-%d")
    os.replace(tmp, path)
    return path


def run_fetch(out_path: Path | None = None) -> Path:
    """Entry point used by Airflow / CLI / KFP: fetch and store the raw CSV."""
    df, source = fetch_btc()
    path = save_csv_atomic(df, out_path or config.raw_path())
    log.info(
        "raw data saved",
        extra={
            "path": str(path),
            "source": source,
            "rows": len(df),
            "first_date": str(df["date"].iloc[0].date()),
            "last_date": str(df["date"].iloc[-1].date()),
        },
    )
    return path
