"""Daily BTC-USD ETL + retraining DAG.

fetch_data -> build_features -> train_and_evaluate -> quality_gate -> register_model -> reload_api

* ML tasks run in /opt/ml-venv through ExternalPythonOperator (@task.external_python):
  Airflow 2.10.5's constraints pin cffi==1.17.1 but yfinance needs cffi>=2.1.1, so the ML
  code lives in its own venv with the SAME pins as the API / training images.
* Every task calls the shared functions in src/ (no logic is duplicated in the DAG).
* quality_gate is a ShortCircuitOperator: if MAPE on the latest 7-day holdout is above
  MAPE_THRESHOLD, register_model and reload_api are SKIPPED and production stays unchanged.
* reload_api tells the Flask service (compose service "api") to re-read the model files.
"""

import os
from datetime import timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.models.param import Param

ML_PYTHON = os.getenv("ML_PYTHON", "/opt/ml-venv/bin/python")

default_args = {
    "owner": "mlops-group",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(minutes=20),
}


@dag(
    dag_id="btc_etl_pipeline",
    description="Daily BTC fetch -> features -> XGBoost retrain -> gated promotion -> API reload",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    catchup=False,  # do not back-fill every day since start_date
    max_active_runs=1,  # runs share data/ and models/ - never run two at once
    default_args=default_args,
    tags=["btc", "mlops", "etl", "xgboost"],
    doc_md=__doc__,
    # Can be overridden per manual run ("Trigger DAG w/ config"), e.g. {"mape_threshold": 1}
    # to demonstrate the gate rejecting a model. Default comes from the MAPE_THRESHOLD env var.
    params={
        "mape_threshold": Param(
            float(os.getenv("MAPE_THRESHOLD", "10")),
            type="number",
            minimum=0,
            description="Promote the new model only if its 7-day holdout MAPE (%) <= this",
        )
    },
)
def btc_etl_pipeline():
    # ---------------- ML tasks: executed by /opt/ml-venv/bin/python ----------------
    # ExternalPythonOperator ships the function's SOURCE to the other interpreter,
    # so all imports must be inside the function body.

    @task.external_python(task_id="fetch_data", python=ML_PYTHON, expect_airflow=False)
    def fetch_data() -> str:
        """Extract: yfinance -> CoinGecko -> Binance (with retries), save raw CSV."""
        from src.etl_fetch import run_fetch
        from src.logging_config import setup_logging

        setup_logging()
        return str(run_fetch())

    @task.external_python(task_id="build_features", python=ML_PYTHON, expect_airflow=False)
    def build_features(raw_path: str) -> str:
        """Transform: clean data, MA7/MA30, volatility, lagged returns, next-day target."""
        from pathlib import Path

        from src import config
        from src.features import run_features
        from src.logging_config import setup_logging

        setup_logging()
        return str(run_features(Path(raw_path), config.features_path()))

    @task.external_python(task_id="train_and_evaluate", python=ML_PYTHON, expect_airflow=False)
    def train_and_evaluate(features_path: str) -> dict:
        """Fit XGBoost, evaluate on the latest 7-day holdout, save a staging candidate."""
        from pathlib import Path

        from src.logging_config import setup_logging
        from src.train_forecast import train

        setup_logging()
        result = train(Path(features_path))
        m = result["metrics"]
        # only small, JSON-friendly values go to XCom
        return {
            "version": result["version"],
            "model_dir": result["model_dir"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "mape": m["mape"],
            "baseline_mape": m["baseline_naive"]["mape"],
            "holdout_start": m["holdout_start"],
            "holdout_end": m["holdout_end"],
        }

    # ---------------- control-flow tasks: run in Airflow's own interpreter ----------------

    @task.short_circuit(task_id="quality_gate")
    def quality_gate(candidate: dict, params=None) -> bool:
        """Continue only if MAPE <= mape_threshold; otherwise skip registration + reload."""
        threshold = float(params["mape_threshold"])  # Airflow injects the run's params
        ok = candidate["mape"] <= threshold
        print(
            f"candidate {candidate['version']}: MAPE={candidate['mape']:.3f}% "
            f"(naive baseline {candidate['baseline_mape']:.3f}%), threshold={threshold}% "
            f"-> {'PASS' if ok else 'FAIL - production model unchanged'}"
        )
        return ok

    @task(task_id="register_model")
    def register_model(candidate: dict) -> str:
        """Promote the candidate to models/<MODEL_STAGE>/ with an atomic directory swap."""
        from pathlib import Path

        from src.registry import promote, prune_staging, read_version

        stage_dir = promote(Path(candidate["model_dir"]))
        prune_staging(keep=5)
        version = read_version(stage_dir)
        print(f"promoted model version {version} to {stage_dir}")
        return version

    @task(task_id="reload_api", retries=3, retry_delay=timedelta(seconds=30))
    def reload_api(version: str) -> dict:
        """POST /reload so the Flask service serves the new model immediately."""
        import requests

        resp = requests.post(
            os.getenv("API_RELOAD_URL", "http://api:5000/reload"),
            headers={"X-API-Key": os.environ["API_KEY"]},
            timeout=30,
        )
        resp.raise_for_status()  # non-2xx -> task fails -> Airflow retries
        body = resp.json()
        print(f"API reload response: {body}")
        if body.get("model_version") != version:
            raise RuntimeError(f"API serves {body.get('model_version')}, expected {version}")
        return body

    candidate = train_and_evaluate(build_features(fetch_data()))
    gate = quality_gate(candidate)
    registered = register_model(candidate)
    gate >> registered
    reload_api(registered)


btc_etl_pipeline()
