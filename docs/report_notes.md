# Report notes: MLOps Assignment 3 (BTC 7-day forecast)

Group: Sami ur Rehman Khan (25L-8005), Muhammad Faizan Aslam (25L-8011)
Repo: https://github.com/samiurkhan10-rgb/btc-forecast-mlops

Raw working log: commands that were run, with trimmed outputs, collected Part by Part.
At the end this file becomes the outline for the 4–6 page PDF report.

---

## Step 0: Repository bootstrap

- `.gitattributes` (`* text=auto eol=lf`) was created **before** any other file, so shell
  scripts, Dockerfiles and YAML keep LF endings even though Git for Windows has `core.autocrlf=true`.
- `.gitignore` / `.dockerignore` exclude `.venv`, `.env`, `data/*`, `models/*`, logs.
- `.env.example` is committed instead of `.env`.
- Requirements are split by audience:
  `requirements-api.txt` (served image) ⊂ `requirements-train.txt` ⊂ `requirements-dev.txt`;
  `requirements-kfp.txt` is only for compiling the pipeline locally.
- `xgboost-cpu` is used instead of `xgboost`. The plain Linux wheel depends on
  `nvidia-nccl-cu12` (~200 MB), which would rule out a small inference image.

```text
$ python -m venv .venv
$ .venv/Scripts/python -m pip install -r requirements-dev.txt
```

---

## Part A: Flask REST API

### Design
- **Model:** XGBoost (`xgboost-cpu` 3.2.0, native `xgb.train`, so no scikit-learn is needed in the
  image). 300 rounds, depth 3, eta 0.03, 2 threads. Training takes about 3 s on the laptop.
- **Target:** the next-day *log return*, not the price. Tree models can't extrapolate beyond
  the target range seen in training, and BTC trends to new highs. Price is rebuilt as
  `close_t * exp(pred)`.
- **Features** (`src/features.py`): 7 lagged log returns, close/MA7−1, close/MA30−1, 7-day and
  30-day return volatility, day of week of the target day. MIN_HISTORY = 31 closes.
- **Recursive 7-day forecast** (`src/forecast.py`). The same function is used by training
  evaluation and by the API, so the metrics measure exactly what the API serves.
- **Evaluation:** the last 7 days are held out, the model is trained on data before them and forecasts
  them recursively. MAE/RMSE/MAPE plus a naive "tomorrow = today" baseline on the same window.
- **Artifacts:** `models/staging/<version>/` → promoted to `models/production/`
  (`model.json`, `metrics.json`, `history.json`, `feature_meta.json`, `version.txt`).
- **Data:** yfinance `BTC-USD`, 3 years of daily candles, CoinGecko fallback. Today's incomplete candle is dropped.

### API endpoints (`app/app.py`, run with `gunicorn app.app:app`)
| Endpoint | Success | Errors |
|---|---|---|
| GET /health | 200 `status, uptime_seconds, model_loaded, model_version, model_stage` | always 200 (probe-safe) |
| POST /predict | 200 `predictions[{date, predicted_close}]` | 400 bad input, 503 no model, 500 |
| GET /metrics | 200 `mae, rmse, mape, baseline_naive, holdout window` | 503 no model |
| POST /reload | 200 `previous_version, model_version, changed` | 401 bad key, 403 key not configured, 503 |

Validation: `forecast_days` must be an int from 1 to 30 (bools, strings and floats rejected); `history` is an array of
31–5000 finite positive numbers; unknown fields are rejected; invalid JSON returns 400; bodies over 256 KB return 413.
Structured JSON logs at LOG_LEVEL. /health requests are logged at DEBUG so probes don't flood the logs.

Multi-worker reload: `/reload` reaches only one gunicorn worker, so every worker also re-reads
`version.txt` (at most every 5 s) and reloads when it changes (`app/model_store.py`).
If loading fails (corrupt files, feature mismatch), the old model keeps serving.

### Verification
```text
$ black .            -> 4 files reformatted, 15 files left unchanged
$ flake8 .           -> (no output, exit 0)
$ python -m pytest
62 passed in 43.99s
app\app.py        97%   app\model_store.py 100%   src\features.py 98%
src\forecast.py  100%   src\train_forecast.py 97% src\etl_fetch.py 96%
TOTAL            483 stmts, 97% coverage
```
All network access is blocked in tests by an autouse fixture (`socket.connect` raises). yfinance
is replaced by a fake module and `requests.get` is monkeypatched for CoinGecko.

```text
$ python -m src.cli all          # real data, local run
fetched BTC data   source=yfinance rows=1096 (2023-11-22 .. 2026-11-21)
model trained      version=20261122T173002Z mae=2429.75 rmse=3345.05 mape=4.02 baseline_mape=3.613
model promoted     version=20261122T173002Z stage=production (threshold 10%)
```

Sample responses (Flask test client against the real model):
```text
GET /health -> 200 {"status":"ok","model_loaded":true,"model_version":"20261122T173002Z",...}
POST /predict {} -> 200
  last_observed 2026-11-21 close 60094.09
  2026-11-22 60434.90 | 11-23 60505.08 | 11-24 60442.52 | 11-25 60153.87
  11-26 59797.68 | 11-27 59361.36 | 11-28 59480.11
POST /predict {"forecast_days":0} -> 400 {"error":"forecast_days must be between 1 and 30","status":400}
GET /metrics -> 200 mae 2429.75, rmse 3345.05, mape 4.02 (naive baseline mape 3.61), holdout 2026-11-15..21
POST /reload (X-API-Key) -> 200 {"reloaded":true,"changed":false,...}
```

**Performance note for the report:** on the 7-day holdout the model (MAPE 4.02%) is slightly
*worse* than the naive persistence baseline (3.61%). Daily BTC returns are close to a random walk,
so naive persistence is a strong baseline at short horizons. The point of this assignment is the
MLOps pipeline. Possible next steps: exogenous features (volume, funding rates), quantile
forecasts, or gating promotion on "beats baseline".

### Issue found and fixed: data fetch failed on a slow network
A later run of `python -m src.cli all` failed with **both** sources down:
```text
ERROR yfinance  $BTC-USD: possibly delisted; no price data found  (period=3y)
WARNING src.etl_fetch  data source failed source=coingecko error=Read timed out. (read timeout=30)
RuntimeError: all data sources failed
```
Diagnosis with curl: every endpoint was reachable, but the **TCP connect alone took about 5 s**:
```text
200 connect=5.09s total=6.61s  query2.finance.yahoo.com
200 connect=4.99s total=7.50s  api.coingecko.com
200 connect=1.47s total=1.91s  api.binance.com
```
Cause: yfinance's default 10 s request timeout expired. yfinance catches the error and returns
an empty frame with a misleading "possibly delisted" message. CoinGecko hit our 30 s read timeout.
Each source got only one attempt.

Fix in `src/etl_fetch.py` (the requirement isn't weakened: a total failure still fails loudly):
- yfinance `timeout=60`; HTTP APIs use `(connect=20 s, read=90 s)`.
- `with_retries()`: 3 attempts per source with exponential backoff (5 s, 15 s),
  configurable with `FETCH_RETRIES` / `FETCH_BACKOFF_SECONDS`.
- Third source **Binance public klines** (BTCUSDT, real daily OHLCV; the assignment lists it).
  Order: yfinance → CoinGecko → Binance.
- 5 new mocked tests (retry backoff, retry exhaustion, yfinance retried, Binance parsing,
  Binance fallback). Result: **67 passed**, flake8 clean.
```text
$ python -m src.cli all     -> source=yfinance rows=1096, version 20261122T195838Z, MAPE 4.02 (33 s)
binance live check: 999 rows 2024-02-27 -> 2026-11-21 last close 60215.77
```

Note: gunicorn doesn't run natively on Windows (it needs `fcntl`). Locally the API is tested with
Flask's test client. Gunicorn itself is shown running in the Docker container in Part B.

### Screenshots for Part A
1. VS Code terminal: `pytest` output with the coverage table and "62 passed".
2. Terminal: `flake8 .` (clean) and `black --check .` ("would be left unchanged").
3. Terminal: `python -m src.cli all` output (fetch → train → promoted with metrics).
4. Sample `/predict` and `/health` JSON. Re-take these against the container in Part B for a nicer screenshot.
