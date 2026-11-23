# Report outline: MLOps Assignment 3 (4–6 page PDF)

**Production MLOps Pipeline for 7-Day BTC-USD Forecasting**
Sami ur Rehman Khan (25L-8005) · Muhammad Faizan Aslam (25L-8011)

How to use this file: each section gives a page budget, the points to write, the figure/screenshot to
place (IDs refer to the screenshot list at the end), and where the numbers come from in
`docs/report_log.md` (the full evidence log). Budget: about 5.5 pages including figures. Keep
screenshots cropped to the relevant part, about 1/3 page each, two per row where possible.

---

## 0. Title block + summary (¼ page)
- Title, names and roll numbers, course, date, links:
  GitHub https://github.com/samiurkhan10-rgb/btc-forecast-mlops ·
  Docker Hub https://hub.docker.com/r/samiurk/btc-forecast-api and https://hub.docker.com/r/samiurk/btc-forecast-train
- 4–5 sentence summary: daily BTC ingestion → XGBoost 7-day recursive forecast → gated promotion →
  Flask/gunicorn API → Docker + GitHub Actions → Airflow + Kubeflow → Kubernetes with HPA. Headline
  results: MAPE ≈ 4 %, 72 tests / 97 % coverage, HPA 2 → 8 pods, 0 failures at 60 concurrent users,
  18 s self-healing with zero dropped requests.

## 1. Architecture (¾ page)
- **Figure 1: architecture diagram.** Export the Mermaid diagram from `README.md`
  (paste it into https://mermaid.live → PNG) or redraw it in draw.io.
- Points to write:
  - One shared `src/` package, four runners: Airflow DAG, K8s Job, KFP container components, API.
  - Data and model flow: sources (yfinance → CoinGecko → Binance) → raw CSV → features CSV →
    `models/staging/<version>` → quality gate → `models/production` → API (compose volume or K8s PVC).
  - Versioned artifacts: model.json, metrics.json, history.json, feature_meta.json, version.txt.
  - The compose and minikube environments never run together (6 GB WSL limit).

## 2. Part A: Model and Flask API (¾ page)
- Model choice: XGBoost on next-day **log returns** (trees can't extrapolate to new price highs), with features
  (7 return lags, MA7/MA30 ratios, 7/30-day volatility, day of week) and a recursive multi-step forecast.
- Evaluation: last 7 days held out, forecast recursively like the API. **Table 1:** MAE 2,529.5 /
  RMSE 3,465.3 / MAPE 4.18 % vs naive baseline MAPE 3.61 %, with one honest sentence on why
  persistence is hard to beat at short horizons.
- API: endpoint table (copy from README), validation rules, error codes 400/401/403/404/405/413/500/503,
  JSON logging, `/health` always 200 for probes, `/predict` 503 without a model.
- Multi-worker hot reload: `/reload` reaches only one gunicorn worker, so every worker checks `version.txt`.
- Testing: 72 offline tests (sockets blocked), parity tests between training and inference features.
- **Screenshots A1** (pytest + coverage), **A2** (sample `/predict` JSON).
- Source: report_log.md → "Part A".

## 3. Part B: Containerization and CI/CD (1 page)
- Multi-stage Dockerfile: builder venv → slim runtime, non-root UID 10001, HEALTHCHECK, `exec gunicorn`,
  requirements copied before code, BuildKit pip cache, build-time import check.
- **Table 2: image sizes** (API 376 MB / 122 MB compressed, train 440 / 145 MB). Explain why the
  < 300 MB target was missed (scipy is a hard dependency of xgboost) and what was removed.
- Compose: postgres + one Airflow `standalone` container + API, shared `./models` (read-only for the API),
  memory limits.
- CI/CD: lint ∥ test → build-and-push (push to main only), Docker Hub secrets, SHA + latest tags,
  GHA layer cache, smoke test of the pushed image.
- **Screenshots B1** (docker build + images), **B2** (compose ps healthy), **B5** (Actions graph green),
  **B7** (Docker Hub tags).
- Box "Issues found": numpy test-folder deletion broke imports → build-time import check; disk full
  crashed the Docker engine → moved its disk to D:.

## 4. Part C: Airflow and Kubeflow (1 page)
- **Figure: Airflow graph view (C2)** of the scheduled run, and **C3** of the gate-fail run (skipped tasks).
- DAG facts: @daily, catchup=False, retries=2 + backoff, max_active_runs=1; task table (copy from log).
- Why ExternalPythonOperator: official constraints pin cffi 1.17.1 vs yfinance's cffi ≥ 2.1.1, so the ML code
  runs in an isolated venv with the same pins as the API (no train/serve version drift).
- ShortCircuit gate with a run-time `mape_threshold` param: PASS run vs FAIL run (register/reload skipped,
  production unchanged). Quote the two log lines (**C4**).
- KFP: 3 container components, Dataset → Dataset → Model + Metrics artifacts, compiled YAML pinned to
  an immutable image SHA, YAML commands verified locally (no cluster). **Screenshot C7** (YAML excerpt).
- Airflow 2.x end-of-life note + what moving to 3.x involves (api-server, Task SDK `airflow.sdk`,
  `logical_date`, separate dag-processor, new constraints). Copy from log.

## 5. Part D: Kubernetes deployment and scaling (1½ pages, the core section)
- Manifest table (copy from log): namespace, ConfigMap/Secret (placeholder in git, real key applied
  from `.env`), PVC 1Gi RWO, Deployment (3 replicas, requests/limits, startup/liveness/readiness,
  non-root, read-only rootfs), ClusterIP + NodePort 30080, HPA v2 2–8 @60 %, Ingress btc-forecast.local, train Job.
- **Screenshot D1**: `kubectl get all,pvc,ingress,hpa`. **D4**: train Job logs → pods switch to the model.
- Access on Windows + docker driver: NodePort via `minikube service --url`; Ingress via hosts file +
  `minikube tunnel` (**D5 / D6**).
- **Figure: HPA scaling (D7)**: side-by-side `get hpa --watch` / `get pods -w`, and the Locust
  charts (**D8**, run 4 report).
- **Table 3: load-test runs 1–4** (copy from log): 90.55 % → 0.43 % → 0.45 % → **0.00 %** failures,
  15.9 → 30.6 req/s, restarts 14 → 0.
- Performance analysis (about 1 paragraph each):
  1. Profiling showed /predict spending ~80 % of its time building one-row pandas DataFrames, so a NumPy path
     made it 22× faster (324 → 15 ms) with identical outputs.
  2. The probe anti-pattern: sync workers + strict liveness → kubelet killed busy pods → cascade;
     fixed with gthread, a liveness probe that catches frozen pods only, and a startupProbe.
  3. The NGINX/gunicorn keep-alive race (60 s vs 2 s) caused the 502s; fixed with `--keep-alive 75`
     via the ConfigMap.
  4. The capacity ceiling is the 2-vCPU node (1,999m at 6–8 pods, ~30 req/s), so HPA alone is not enough;
     a real cluster also needs the Cluster Autoscaler. CPU % is relative to the 100m request.
- Limitation: RWO PVC works only on a single node, so production would need RWX storage or an object store.

## 6. Part E: Resilience and end-to-end verification (¾ page)
- **E1 chaos test:** timeline table (delete 06:06:46.0 → replacement created +0.9 s → Running +5.2 s →
  Ready +16.1 s; back to 2/2 in 18.1 s), **66/66 requests 200 during the test**. Note the first,
  discarded attempt (HPA scale-down interfered). **Screenshots E1, E2.**
- **E2 end-to-end:** manual DAG run `manual__e2e_verification` → version 20261122T222954Z →
  **20261123T011108Z**; the same artifact promoted into the K8s PVC; NodePort and Ingress `/predict`
  return that version; all CI runs green. **Screenshots E3–E7.**

## 7. Lessons learned and limitations (½ page)
- Load testing found three production bugs that unit tests could not (inference cost, probe
  design, keep-alive race).
- Resource-constrained engineering: single Airflow container, isolated venvs, CPU-only xgboost, cached
  images, disk relocation, never running both stacks.
- Limitations: Airflow 2.x EOL, RWO PVC / single node, image size > 300 MB, no Kubeflow cluster,
  model ≈ naive baseline (BTC is close to a random walk). Future work: CronJob or KFP schedule
  on K8s, an object-store model registry (MLflow/S3), Cluster Autoscaler, beat-the-baseline gate.

---

## Rubric checklist

| Item | Requirement | Covered by (file) | Evidence (log section / screenshot) |
|---|---|---|---|
| A1 | GET /health (status, uptime, model version) | `app/app.py` | Part A sample responses · A2 |
| A1 | POST /predict (forecast_days default 7, custom history, 7 dates + prices) | `app/app.py`, `src/forecast.py` | Part A, E2 NodePort/Ingress output · A2, E6 |
| A1 | GET /metrics (MAE, RMSE, MAPE) | `app/app.py`, `src/train_forecast.py` | Part A `/metrics` output |
| A2 | Validation, try/except, 400/500 JSON errors | `app/app.py` (`parse_predict_payload`, error handlers) | `tests/test_api.py` (12 invalid-payload cases, 404/405/500) |
| A3 | Gunicorn WSGI + pytest for endpoints | `app/Dockerfile` CMD, `tests/test_api.py` | Part B container run (gunicorn workers) · A1 |
| B1 | Multi-stage, non-root, slim, layer caching, small | `app/Dockerfile` | Part B image sizes, Issue 1 · B1 |
| B2 | docker-compose: API + Airflow + Postgres | `docker-compose.yml`, `docker/Dockerfile.airflow` | Part B verification · B2, B4 |
| B3 | CI on push/PR: flake8/black + pytest | `.github/workflows/ci_cd.yml` (lint, test) | Part B CI table · B5 |
| B4 | Docker Hub login via secrets, SHA + latest, push | `ci_cd.yml` (build-and-push) | Docker Hub tags · B6, B7 |
| C1 | DAG: fetch task | `dags/btc_etl_pipeline.py`, `src/etl_fetch.py` | Part C run 1 log |
| C1 | DAG: preprocess + features (MA7/MA30, volatility) | `src/features.py` | Part C · C2 |
| C1 | DAG: retrain + evaluate on 7-day window + register if threshold passes | `src/train_forecast.py`, `src/registry.py`, ShortCircuit gate | Part C runs 1 & 2 · C2, C3, C4 |
| C1 | DAG: API reload trigger (HTTP) | `reload_api` task → `POST /reload` | Part C run 1 · C5 |
| C2 | KFP SDK components (ingest, features, train) compiled to YAML | `kubeflow/btc_kfp_pipeline.py`, `.yaml` | Part C KFP · C7 |
| D1 | ConfigMap (MODEL_STAGE, LOG_LEVEL, PORT) + Secret as env | `k8s/configmap-secret.yaml`, `deployment.yaml` envFrom | Part D bring-up · D2, D3 |
| D2 | PVC mounted at /app/models, persists across restarts | `k8s/pvc.yaml`, `train-job.yaml` | Part D + E1 (new pod serves the PVC model), node restart · D4 |
| D3 | 3 replicas, liveness (15/10), readiness, requests/limits | `k8s/deployment.yaml` | Part D manifest table · D1, D2 |
| D4 | ClusterIP + NodePort, 80 → 5000 | `k8s/service.yaml` | Part D external access · D5 |
| D5 | HPA min 2 / max 8 / CPU 60 % + load test with `get hpa --watch` | `k8s/hpa.yaml`, `loadtest/locustfile.py` | Part D runs 1–4 · D7, D8 |
| D6 | NGINX Ingress host btc-forecast.local | `k8s/ingress.yaml` | Part D + E2 Ingress · D6 |
| E1 | Delete a pod, document self-healing | n/a (kubectl) | Part E E1 timeline · E1, E2 |
| E2 | Manual DAG run → model updated, CI green, live K8s /predict | DAG + `kubectl cp`/promote | Part E E2 · E3–E7 |
| E3 | 4–6 page PDF with screenshots, CLI logs, architecture diagram, load-test output, performance analysis | this outline + `docs/report_log.md` | all |
| Deliverables | GitHub repo · public Docker Hub images · PDF | repo, `README.md` | Docker Hub links (public: `is_private = False`) |

---

## Screenshot list (IDs used above)

| ID | What | Taken? |
|---|---|---|
| A1 | `pytest` output with coverage table (72 passed) | ☑ |
| A2 | `flake8` / `black --check` clean + `python -m src.cli all` output | ☑ |
| B1 | `docker build` tail (`imports ok`) + `docker images` sizes | ☑ |
| B2 | `docker compose ps` (3 healthy) + `docker stats --no-stream` | ☑ |
| B3 | `/health` + `/predict` JSON from the compose API | ☑ |
| B4 | Airflow UI home (v2.10.5) | ☑ |
| B5 | GitHub Actions run graph: lint, test → build-and-push green | ☑ |
| B6 | build-and-push job steps (login, pushes, smoke test) | ☑ |
| B7 | Docker Hub Tags tab (SHA + latest) for both repos | ☑ |
| C1 | Airflow DAG list with btc_etl_pipeline | ☑ |
| C2 | Graph view: scheduled run, all 6 green | ☑ |
| C3 | Graph view: `manual__gate_fail_demo`, register/reload skipped | ☑ |
| C4 | quality_gate logs (PASS and FAIL lines) | ☑ |
| C5 | reload_api log (API reload response) | ☑ |
| C6 | `/health` before/after the DAG (version change) | ☑ |
| C7 | `btc_kfp_pipeline.yaml` excerpt + compile command | ☑ |
| D1 | `kubectl -n btc-forecast get all,pvc,ingress,hpa` | ☑ |
| D2 | `kubectl describe deploy btc-api` (probes, resources, env) | ☑ |
| D3 | ConfigMap YAML + `get secret` | ☑ |
| D4 | train Job status + logs | ☑ |
| D5 | NodePort URL + `/predict` | ☑ |
| D6 | Ingress `/predict` (btc-forecast.local) + tunnel window | ☑ |
| D7 | side-by-side `get hpa --watch` / `get pods -w` during Locust (2 → 8) | ☑ |
| D8 | Locust HTML report charts (run4_report.html) | ☑ |
| E1 | pod deletion with timestamps + `get pods -w` | ☑ |
| E2 | `get events` (SuccessfulCreate / Killing) | ☑ |
| E3 | Airflow `manual__e2e_verification` grid/details | ☑ |
| E4 | `/health` before/after the manual run | ☐ |
| E5 | `kubectl cp` + promote + both pods' new version | ☐ |
| E6 | NodePort + Ingress `/predict` with version 20261123T011108Z | ☐ |
| E7 | GitHub Actions run list, all green | ☐ |
