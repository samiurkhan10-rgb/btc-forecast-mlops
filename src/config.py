"""Central configuration: every path / setting comes from an env var with a sane default.

Values are read through functions (not module constants) so tests and different
runtimes (local, Docker, Airflow, K8s) can override them via environment variables.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TICKER = "BTC-USD"
RAW_FILENAME = "btc_usd_raw.csv"
FEATURES_FILENAME = "btc_usd_features.csv"

# Model artifact file names (one directory per model version)
MODEL_FILE = "model.json"
METRICS_FILE = "metrics.json"
HISTORY_FILE = "history.json"
META_FILE = "feature_meta.json"
VERSION_FILE = "version.txt"

HOLDOUT_DAYS = 7  # evaluation window = the last 7 days (assignment: "latest 7-day test window")


def data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))


def models_dir() -> Path:
    return Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models")))


def model_stage() -> str:
    return os.getenv("MODEL_STAGE", "production")


def mape_threshold() -> float:
    """Quality gate (percent). A newly trained model is only promoted if MAPE <= this."""
    return float(os.getenv("MAPE_THRESHOLD", "10"))


def raw_path() -> Path:
    return data_dir() / "raw" / RAW_FILENAME


def features_path() -> Path:
    return data_dir() / "processed" / FEATURES_FILENAME
