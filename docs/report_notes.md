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

---

## Part B: Containerization & CI/CD

### Files
- `app/Dockerfile`: multi-stage. The builder stage creates `/opt/venv` from `requirements-api.txt` only;
  the runtime stage is `python:3.10-slim` + that venv + only the 5 `src/` modules the API imports + `app/`.
  Non-root user `app` (UID 10001), `HEALTHCHECK` on `/health`, gunicorn as PID 1 via `exec`
  (receives SIGTERM → graceful shutdown), 2 sync workers, `--worker-tmp-dir /dev/shm`.
  Layer caching: requirements are copied before code, and a BuildKit pip cache mount means re-builds don't re-download wheels.
- `docker/Dockerfile.train`: same pattern, `requirements-train.txt`, `ENTRYPOINT python -m src.cli`, default `all`.
- `docker/Dockerfile.airflow`: `apache/airflow:slim-2.10.5-python3.10` + `apache-airflow[postgres]` with the official
  constraints file + a separate `/opt/ml-venv` for the ML code (see the dependency conflict below).
- `docker-compose.yml`: three services (`postgres`, `airflow` running `airflow standalone` with LocalExecutor,
  `api`). `./models` is shared: Airflow writes it, the API mounts it read-only. Memory limits keep the laptop usable.
  The Airflow reload task calls `http://api:5000/reload`, using the compose service name.
- `.github/workflows/ci_cd.yml`: `lint` (flake8 + black) and `test` (pytest + coverage, reports uploaded) on
  push/PR, pip caching on Python 3.10. `build-and-push` runs only on push to main after both pass: buildx,
  Docker Hub login from secrets, API and train images tagged `${{ github.sha }}` and `latest`, GHA layer
  cache, then a smoke test of the pushed API image (`/health` 200 with model_loaded=false, `/predict` 503).

### Image sizes
| Image | Uncompressed | Compressed (registry) |
|---|---|---|
| samiurk/btc-forecast-api | 376 MB | 122 MB |
| samiurk/btc-forecast-train | 440 MB | 145 MB |
| btc-forecast-airflow (local only) | 1.2 GB (386 MB is /opt/ml-venv) | n/a |

Breakdown of the API venv (248 MB): scipy 85 MB + scipy.libs 27 MB, pandas 38 MB, numpy 29 MB +
numpy.libs 27 MB, xgboost 18 MB. **scipy is a hard dependency**: `xgboost/core.py` does
`import scipy.sparse` at import time, so it can't be dropped. The < 300 MB target was not met.
Size was accepted as is: it's already minimal for XGBoost+pandas on python:3.10-slim (no
compilers, no pip/setuptools, no test suites, CPU-only xgboost wheel avoids the ~200 MB NCCL library).

### Issue 1: first image built "successfully" but could not import xgboost
To shrink the image, every `tests/` directory in site-packages was deleted. That also removed
`numpy/_core/tests`, which `numpy.testing` needs at import time, and scipy imports `numpy.testing`:
```text
ModuleNotFoundError: No module named 'numpy._core.tests'
```
Fix: exclude numpy from the cleanup, and the builder stage now ends with
`python -c "import xgboost, scipy.sparse, pandas, numpy, flask, gunicorn"`, so a broken
image fails the **build** instead of failing in production. Lesson: size optimisations need a runtime check.

### Issue 2: Docker engine crashed mid-build (disk full)
```text
ERROR: Could not install packages due to an OSError: [Errno 5] Input/output error
ERROR: failed to build: ... error reading from server: EOF
```
Cause: C: had 2.1 GB free. Docker Desktop's WSL disk (`docker_data.vhdx`, 10.6 GB) lives on C: and
could not grow, so the VM got I/O errors and the engine hung. Fix: moved the disk image to D:
(Docker Desktop → Settings → Resources → Advanced → Disk image location), which kept all images,
including minikube's node image. C: free space went from 2.1 GB to 16 GB.

### Issue 3: Airflow constraints vs yfinance dependency conflict
```text
ERROR: Cannot install yfinance ... conflict is caused by:
    curl-cffi 0.17.0 depends on cffi>=2.1.1
    The user requested (constraint) cffi==1.17.1
```
Airflow 2.10.5's official constraints pin `cffi==1.17.1` (used by Airflow's crypto stack), while
yfinance 1.8 → curl_cffi 0.17 needs cffi ≥ 2.1.1. Options considered: override the pin (breaks the
"official constraints" guarantee), downgrade yfinance (old versions break when Yahoo changes its API), or
**isolate the ML code** (chosen, and Airflow's documented pattern for conflicting deps): Airflow core
is installed strictly with constraints, and the ML tasks run in `/opt/ml-venv` through
`ExternalPythonOperator`. That venv uses the same `requirements-train.txt` as the train/API images,
so the model is trained with exactly the library versions the API serves with.

### Verification
```text
$ docker build -f app/Dockerfile -t samiurk/btc-forecast-api:local .
#12 428.2 imports ok                     <- build-time import check
$ docker run -d -p 5000:5000 -e API_KEY=demo -v ./models:/app/models:ro samiurk/btc-forecast-api:local
[INFO] Starting gunicorn 23.0.0 / Listening at: http://0.0.0.0:5000 / Booting worker x2
GET  /health  -> {"status":"ok","model_loaded":true,"model_version":"20261122T200335Z",...}
POST /predict -> 7 predictions, 2026-11-22 60434.9 ... 2026-11-28 59480.11 (identical to the Windows run)
docker top: 3 gunicorn processes, user 10001   docker inspect Health: healthy

$ docker compose up -d --wait
Container btc-forecast-postgres-1 Healthy / btc-forecast-api-1 Healthy / btc-forecast-airflow-1 Healthy
docker stats: airflow 727 MiB / 2 GiB, postgres 84 MiB / 256 MiB, api 155 MiB / 512 MiB
airflow -> curl http://api:5000/health            -> 200 (service-name DNS works)
airflow -> curl -X POST -H "X-API-Key: $API_KEY" $API_RELOAD_URL -> {"reloaded":true,...}
ML venv: xgboost 3.2.0 | pandas 2.2.3 | numpy 2.2.6 | yfinance 1.8.0   Airflow core: 2.10.5, cffi 1.17.1
```

GitHub Actions run 45075612809 (commit 3f7999f), all green, about 3 minutes total:
```text
Lint (flake8 + black)            22:04:27 -> 22:04:42  success
Unit tests (pytest + coverage)   22:04:27 -> 22:05:09  success
Build & push Docker images       22:05:11 -> 22:07:33  success (login, api push, train push, smoke test)
Docker Hub (public):
  samiurk/btc-forecast-api    tags 3f7999ff8887..., latest   121.9 MB compressed
  samiurk/btc-forecast-train  tags 3f7999ff8887..., latest   145.1 MB compressed
```
Links: https://github.com/samiurkhan10-rgb/btc-forecast-mlops/actions ·
https://hub.docker.com/r/samiurk/btc-forecast-api · https://hub.docker.com/r/samiurk/btc-forecast-train

KFP YAML final pin: `docker.io/samiurk/btc-forecast-train:8858a4392b3a7f7ceeab9d46beaa554c305d1b82` (image built by CI for commit 8858a43, includes the XDG_CACHE_HOME fix).

### Screenshots for Part B
1. Terminal: `docker build` of the API image (end of output, including `imports ok`) and `docker images` with the sizes.
2. Terminal: `docker compose ps` showing all 3 services **healthy**, plus `docker stats --no-stream`.
3. Browser: `http://localhost:5000/health` and a `/predict` response from the container (the Part A JSON screenshot).
4. Browser: Airflow UI login page / home at `http://localhost:8080` (no DAGs yet; the DAG comes in Part C).
5. GitHub → Actions → run for commit 3f7999f: the graph with lint, test → build-and-push all green.
6. GitHub → the build-and-push job expanded, showing "Log in to Docker Hub", both pushes and "Smoke test" green.
7. Docker Hub → `samiurk/btc-forecast-api` Tags tab showing the SHA tag and `latest` (and the train repo).

---

## Part C: Airflow ETL DAG & Kubeflow pipeline

### Airflow DAG (`dags/btc_etl_pipeline.py`)
`@daily`, `start_date 2026-11-01`, `catchup=False`, `max_active_runs=1`,
`retries=2` with exponential backoff, 20 min `execution_timeout`.

| Task | Operator | Runs in | What it does |
|---|---|---|---|
| fetch_data | ExternalPythonOperator | /opt/ml-venv | `src.etl_fetch.run_fetch()`: yfinance → CoinGecko → Binance, retries |
| build_features | ExternalPythonOperator | /opt/ml-venv | `src.features.run_features()`: MA7/MA30, volatility, lagged returns, target |
| train_and_evaluate | ExternalPythonOperator | /opt/ml-venv | `src.train_forecast.train()`: XGBoost + recursive 7-day holdout metrics → staging |
| quality_gate | ShortCircuitOperator | Airflow | pass if MAPE ≤ `mape_threshold` (DAG param, default env MAPE_THRESHOLD=10) |
| register_model | PythonOperator | Airflow | `src.registry.promote()`: atomic swap into models/production, prune staging |
| reload_api | PythonOperator | Airflow | `POST http://api:5000/reload` with X-API-Key; verifies the served version |

If the gate fails, `register_model` and `reload_api` are **skipped** (not failed), and the API keeps
serving the previous model. Only small JSON values go through XCom; data and models go through the
shared volume.

### Run 1: scheduled run (unpaused → scheduler created the 2026-11-21 interval run)
```text
run scheduled__2026-11-21T00:00:00+00:00   success   22:28:19 -> 22:30:24 (~2 min)
fetch_data         success  fetched BTC data source=yfinance rows=1096 (2023-11-22..2026-11-21)
build_features     success
train_and_evaluate success  model trained version=20261122T222954Z mae=2529.52 rmse=3465.34 mape=4.185
quality_gate       success  candidate 20261122T222954Z: MAPE=4.185% (naive baseline 3.613%), threshold=10.0% -> PASS
register_model     success  promoted model version 20261122T222954Z to /opt/***/models/production
reload_api         success  API reload response: {'reloaded': True, 'model_version': '20261122T222954Z', ...}
$ curl localhost:5000/health -> "model_version":"20261122T222954Z"   (was 20261122T200335Z)
```
Notes:
- `changed: False` in the reload response: the gunicorn worker's own 5 s `version.txt` check had
  already picked up the new model before `/reload` arrived. This is the multi-worker design from Part A working;
  the task asserts that the API serves exactly the registered version.
- `/opt/***/` is Airflow's secret masker hiding the word "airflow" because it equals the local
  Postgres password (local-dev credentials only).

### Run 2: gate rejection demo (manual trigger with config)
```text
$ airflow dags trigger btc_etl_pipeline -r manual__gate_fail_demo -c '{"mape_threshold": 1}'
fetch_data success | build_features success | train_and_evaluate success | quality_gate success
register_model skipped | reload_api skipped                      run state: success
candidate 20261122T223422Z: MAPE=4.185% (naive baseline 3.613%), threshold=1.0% -> FAIL - production model unchanged
$ curl localhost:5000/health -> "model_version":"20261122T222954Z"   (unchanged)
```

### Kubeflow pipeline (`kubeflow/btc_kfp_pipeline.py` → `btc_kfp_pipeline.yaml`)
- KFP v2 SDK 2.18.1, three `@dsl.container_component`s: `ingest_data` → `generate_features` →
  `train_model`, each running `python -m src.cli …` in `samiurk/btc-forecast-train`. That's the same
  `src/` code as Airflow and the K8s Job; the image contains no kfp (KFP's launcher handles artifacts).
- Artifacts: `raw_data: Dataset` → `features: Dataset` → `model: Model` + `metrics: Metrics`.
- Ingest has caching disabled (new data daily) and 3 retries with backoff. Every step requests 250m CPU / 256Mi
  and is limited to 1 CPU / 1Gi.
- The YAML is pinned to an **immutable SHA tag** of the train image, not `:latest`, for reproducibility.
- Compiled locally in a separate venv (`.venv-kfp`) so kfp's pins can't affect the project venv:
```text
$ python kubeflow/btc_kfp_pipeline.py --image-tag <sha>
compiled kubeflow/btc_kfp_pipeline.yaml with image docker.io/samiurk/btc-forecast-train:<sha>
```
- No Kubeflow cluster was installed (too heavy for the laptop). Instead, the exact commands from the
  compiled YAML were run in the train image with a shared volume as artifact storage:
```text
ingest_data       -> /kfp/ingest/raw_data
generate_features -> /kfp/features/features
train_model       -> /kfp/train/model/20261122T224522Z/{model.json,metrics.json,...}, /kfp/train/metrics (mape 4.185)
```

### Issue 4: compiled YAML ignored the requested image tag
The first compile still said `:latest`: KFP builds a container component's `ContainerSpec` when the
decorator runs at import time, so changing a module-level variable afterwards had no effect. Fix:
a `build_pipeline(image)` factory defines the components after the image is known. A CI test now
fails if the YAML uses `:latest` or more than one image.

### Issue 5: yfinance cache warnings in the non-root container
`Permission denied: '/home/app'`: the non-root user has no home directory, so yfinance could not
cache Yahoo's timezone/cookie data and re-downloaded it on every run. Fix: `XDG_CACHE_HOME=/tmp/.cache`
in `Dockerfile.train`.

### CI coverage of orchestration code
`tests/test_pipelines.py` checks the DAG with `ast` (schedule `@daily`, `catchup=False`, retries, the 6
task ids, 1 short-circuit + 3 external-python tasks) and the KFP YAML as text (3 components,
Dataset/Model/Metrics, single SHA-pinned image), without installing Airflow or kfp in CI.
70 tests pass.

### Airflow 2.10.5 end-of-life and moving to 3.x
Airflow 2.x is past end-of-life. 2.10.5 was used because it is lighter (a single `airflow standalone`
container fits the 6 GB WSL limit) and widely documented. Moving to Airflow 3.x would involve:
- `airflow webserver` → `airflow api-server` (FastAPI). The DAG processor becomes a mandatory separate process.
- Tasks talk to the metadata DB through the Task Execution API / Task SDK (`airflow.sdk`), not directly.
  Imports move from `airflow.decorators`/`airflow.models` to `airflow.sdk` (`dag`, `task`, `Param`).
- Removed context variables such as `execution_date` (use `logical_date`). `schedule_interval` is gone
  (this DAG already uses `schedule=`). `catchup` defaults to False.
- Provider packages and the constraints file change. Re-check the cffi/yfinance conflict; the isolated
  `/opt/ml-venv` design carries over unchanged.
- Run `airflow db migrate` and the `ruff` AIR3xx rules to find incompatible DAG code.

### Screenshots for Part C
1. Airflow UI → DAGs list showing `btc_etl_pipeline` (schedule @daily, tags, 2 successful runs).
2. Airflow UI → btc_etl_pipeline → **Graph** view of the scheduled run: all 6 tasks green.
3. Airflow UI → Graph view of `manual__gate_fail_demo`: register_model / reload_api **skipped** (pink).
4. Airflow UI → quality_gate task → Logs for each run (the PASS and FAIL lines).
5. Airflow UI → reload_api → Logs showing the API reload response.
6. Terminal: `curl localhost:5000/health` before/after showing the version change.
7. VS Code: `kubeflow/btc_kfp_pipeline.yaml` (components + SHA-pinned image) and the compile command output.

---

## Part D: Kubernetes deployment & scaling (minikube v1.39, Kubernetes v1.34.0, docker driver)

Sequence: `docker compose down` → `minikube start` (all addon images cached). Note: `minikube addons list`
shows ingress as "disabled", but the ingress-nginx controller from the earlier `addons enable` is installed
and Running (IngressClass `nginx` present), so it was left untouched.

### Manifests (`k8s/`, namespace `btc-forecast`)
| File | Content |
|---|---|
| namespace.yaml | `btc-forecast` namespace |
| configmap-secret.yaml | ConfigMap `MODEL_STAGE`, `LOG_LEVEL`, `PORT`, `MODELS_DIR`, `MAPE_THRESHOLD`, `GUNICORN_CMD_ARGS`; Secret `API_KEY`, `COINGECKO_API_KEY` (**placeholder in git**; the real key is applied from `.env` with `kubectl create secret generic ... --dry-run=client -o yaml`, piped to `kubectl apply -f -`) |
| pvc.yaml | `btc-models-pvc` 1Gi **RWO**, storageClass `standard`, mounted at `/app/models` |
| deployment.yaml | 3 replicas, CI image pinned by commit SHA, `IfNotPresent`, requests 100m/128Mi, limits 500m/512Mi, startup + liveness (15 s/10 s) + readiness probes on `/health`, `envFrom` ConfigMap + Secret, non-root 10001, read-only root FS, all capabilities dropped, PVC mounted read-only |
| service.yaml | `btc-api` ClusterIP 80→5000 and `btc-api-nodeport` NodePort 80→5000 (nodePort 30080) |
| hpa.yaml | autoscaling/v2, min 2, max 8, CPU 60%; scale-up 2 pods / 30 s, scale-down 1 pod / 30 s after 120 s |
| ingress.yaml | `ingressClassName: nginx`, host `btc-forecast.local` → `btc-api:80` |
| train-job.yaml | Job running the CI train image `all` → writes the model (and `data/`) into the PVC |

Image delivery: the CI-built images were pulled once into the node
(`minikube ssh -- sudo crictl pull docker.io/samiurk/btc-forecast-{api,train}:<sha>`, about 4 min on the
slow link). With `imagePullPolicy: IfNotPresent` no further downloads happen. `minikube image load`
was not needed.

**Limitation (RWO PVC):** ReadWriteOnce works because minikube has a single node: the train Job and
all API replicas are on the same node. A multi-node cluster needs ReadWriteMany storage (NFS/CephFS/EFS)
or an object store (S3/MinIO) for model artifacts.

### Bring-up
```text
$ kubectl apply -f k8s/   (namespace, configmap-secret, pvc, deployment, service, hpa, ingress)
deployment "btc-api" successfully rolled out          3/3 Running, PVC Bound (pvc-f323c763...)
pod: uid=10001(app); touch /app/x -> Read-only file system; API_KEY set: yes
GET /health  -> {"status":"ok","model_loaded":false,"model_version":null}   (empty PVC, probes pass)
POST /predict -> 503 {"error":"model not loaded yet - run the training pipeline first"}

$ kubectl apply -f k8s/train-job.yaml
job.batch/btc-train   Complete   1/1   51s
fetched BTC data source=yfinance rows=1096 | model trained version=20261122T234248Z mape=4.185 | promoted: true
all 3 replicas -> "model_loaded":true,"model_version":"20261122T234248Z"   (no restart, 5 s version check)
PVC: /app/models/{production,staging,data}
```
Rolling updates later replaced every pod; the new pods loaded the same model straight from the PVC
(persistence across pod restarts, D2).

### External access
```text
NodePort: minikube service btc-api-nodeport -n btc-forecast --url  -> http://127.0.0.1:29385
  (docker driver on Windows: the node IP 192.168.49.2 is not routable from Windows, so minikube tunnels it)
  GET /health 200, POST /predict 200 (7 predictions, 2026-11-22 60670.17 ... 2026-11-28 60765.99)
Ingress: hosts file "127.0.0.1 btc-forecast.local" + `minikube tunnel` (Administrator)
  curl http://btc-forecast.local/health -> HTTP 200 in 0.05 s
  Host: other.local -> HTTP 404 (host-based routing)
  ingress-nginx log: "POST /predict HTTP/1.1" 200 ... [btc-forecast-btc-api-80] 10.244.0.6:5000
```

### HPA load tests (Locust, 60 users, spawn 5/s, 4 min, via the Ingress)
`.venv\Scripts\locust -f loadtest\locustfile.py --host http://btc-forecast.local --headless -u 60 -r 5 -t 4m`
(/predict weight 3 with 7/14/30-day horizons, /health weight 1)

| Run | Change | Requests | Failures | req/s | Median /predict | Pod restarts | HPA |
|---|---|---|---|---|---|---|---|
| 1 | baseline | 3,769 | **90.55 %** (503/502) | 15.9 | 78 ms* | 14 | 2→4→6→8 |
| 2 | NumPy fast path + gthread + softer liveness | 7,386 | 0.43 % (502) | 30.9 | 1,100 ms | 1 | 2→3→5→7→8 |
| 3 | + startupProbe | 7,057 | 0.45 % (502) | 29.5 | 1,200 ms | **0** | 2→4→6→8 |
| 4 | + gunicorn keep-alive 75 s | 7,315 | **0.00 %** | 30.6 | 1,100 ms | **0** | 2→4→6→8 |

\* Run 1's median is low only because most requests failed fast with 503.

Run 4 HPA timeline (`kubectl get hpa --watch`):
```text
05:32:52  cpu: 2%/60%    replicas 2
05:33:22  cpu: 65%/60%   replicas 2
05:34:23  cpu: 467%/60%  replicas 2
05:34:38  cpu: 467%/60%  replicas 4
05:35:10  cpu: 467%/60%  replicas 6
05:35:43  cpu: 402%/60%  replicas 8   (maxReplicas)
05:36:16  cpu: 204%/60%  replicas 8
node CPU (kubectl top node): 407m -> 747m -> 1482m -> 1854m -> 1999m   (2-vCPU minikube limit)
```
After the test the HPA scaled back down to minReplicas=2 (one pod / 30 s after the 120 s window).

### Problems found by the load test and their fixes
1. **Slow inference.** `/predict` (7 days) took about 420 ms in the pod. cProfile: about 80 % went into building a one-row
   pandas DataFrame per recursive step (47 ms/step) vs 5 ms for XGBoost. Fix: `next_step_vector()` in
   NumPy (7-day forecast 324 ms → 15 ms, 30-day 66 ms locally), with identical predictions, enforced by two
   new parity tests (every training row, and a 14-step forecast vs a pandas reference loop).
2. **Liveness probes killing busy pods.** Sync gunicorn workers serve one request at a time, so `/health`
   probes queued behind `/predict`, timed out after 3 s, and the kubelet restarted healthy-but-busy
   pods. Events: `Liveness probe failed: context deadline exceeded`, exit code 0 (graceful kill, no OOM).
   The surviving pods got more load, so the failure cascaded and NGINX returned 503 (no ready endpoints).
   Fix: `gthread` (still 2 workers, 4 threads each); liveness timeout 5 s × 5 failures ("frozen", not
   "busy"); readiness stays strict (busy pods leave the endpoints briefly instead of restarting).
3. **Pods restarted while still booting.** A pod created on the saturated node booted slowly and liveness got
   `connection refused`. Fix: `startupProbe` (up to 120 s); liveness/readiness start after it passes.
4. **Rare 502s (0.45 %): keep-alive race.** NGINX log: `recv() failed (104: Connection reset by peer) while
   reading response header from upstream`, with no errors in gunicorn. ingress-nginx keeps upstream keepalive
   connections for 60 s, gunicorn's default keep-alive is 2 s, so gunicorn closed connections NGINX was about
   to reuse. Fix: `--keep-alive 75` via `GUNICORN_CMD_ARGS` in the ConfigMap (no rebuild needed; same default
   added to the Dockerfile). Verified: a connection idle for 70 s is still reused. Result: 0 failures.

### Performance analysis
- Capacity is **CPU-bound at the node**: node CPU reaches 1,999m (the 2-vCPU minikube limit) with 6–8 pods.
  Pods 6–8 add almost no throughput (about 30 req/s ceiling). The HPA scales pods correctly, but a real cluster
  would also need the Cluster Autoscaler to add nodes.
- 60 users is above the cluster's capacity, so requests queue (median 1.1 s, p95 4.1 s for /predict).
  Each /predict costs about 15–70 ms of CPU depending on the horizon. Under capacity the latency is that service time.
- CPU utilisation is measured against the 100m **request**: 467 % ≈ 467m per pod, close to the 500m limit
  (pods were CPU-throttled), which is why the HPA immediately asks for the maximum.
- HPA vs Deployment: the Deployment declares 3 replicas, the HPA minimum is 2. Re-applying
  `deployment.yaml` briefly resets to 3 until the HPA takes over again. In production, drop `replicas`
  from the manifest once an HPA manages it.

### Screenshots for Part D
1. `kubectl -n btc-forecast get all,pvc,ingress,hpa` (pods, services, PVC Bound, HPA, ingress).
2. `kubectl -n btc-forecast describe deploy btc-api` (probes, resources, env from ConfigMap/Secret, image SHA).
3. `kubectl -n btc-forecast get cm btc-api-config -o yaml` and `get secret btc-api-secret` (no values shown).
4. Train Job: `kubectl -n btc-forecast get job` + `logs job/btc-train` (fetched → trained → promoted).
5. `minikube service btc-api-nodeport -n btc-forecast --url` + a `/predict` call to that URL.
6. `Invoke-RestMethod http://btc-forecast.local/predict` through the Ingress + the tunnel window.
7. **Side-by-side**: `kubectl get hpa --watch` and `kubectl get pods -w` during a Locust run (2 → 8 pods).
8. Locust HTML report (`loadtest/results/run4_report.html`) charts: RPS, response times, users.
9. Optional "before" evidence: run 1 report/failure table and `kubectl get events` with the liveness failures.
