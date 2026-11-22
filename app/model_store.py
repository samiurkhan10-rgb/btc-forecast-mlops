"""Loads the served model from <MODELS_DIR>/<MODEL_STAGE>/ and keeps it in memory.

Why the version check: gunicorn runs several worker processes and POST /reload
only reaches ONE of them. Every worker therefore also re-reads version.txt (at most
every CHECK_INTERVAL seconds) and reloads itself when the version has changed.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src import config
from src.features import FEATURE_COLUMNS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    booster: xgb.Booster
    version: str
    metrics: dict
    closes: list
    last_date: pd.Timestamp
    meta: dict = field(default_factory=dict)


class ModelStore:
    CHECK_INTERVAL = 5.0  # seconds between version.txt checks per worker

    def __init__(self, models_dir: Path, stage: str):
        self.stage_dir = Path(models_dir) / stage
        self._lock = threading.Lock()
        self._model: LoadedModel | None = None
        self._last_check = 0.0

    @property
    def model(self) -> LoadedModel | None:
        return self._model

    def _read_disk_version(self) -> str | None:
        path = self.stage_dir / config.VERSION_FILE
        try:
            return path.read_text().strip() or None
        except OSError:
            return None

    def load(self) -> bool:
        """(Re)load from disk. On any failure the previously loaded model is kept."""
        with self._lock:
            version = self._read_disk_version()
            if version is None:
                log.warning("no model found", extra={"stage_dir": str(self.stage_dir)})
                return self._model is not None
            try:
                meta = json.loads((self.stage_dir / config.META_FILE).read_text())
                if meta.get("feature_columns") != FEATURE_COLUMNS:
                    raise ValueError("model was trained with different features (skew guard)")
                booster = xgb.Booster()
                booster.load_model(str(self.stage_dir / config.MODEL_FILE))
                metrics = json.loads((self.stage_dir / config.METRICS_FILE).read_text())
                history = json.loads((self.stage_dir / config.HISTORY_FILE).read_text())
                self._model = LoadedModel(
                    booster=booster,
                    version=version,
                    metrics=metrics,
                    closes=[float(c) for c in history["closes"]],
                    last_date=pd.Timestamp(history["dates"][-1]),
                    meta=meta,
                )
                log.info("model loaded", extra={"model_version": version})
                return True
            except Exception:  # noqa: BLE001 - keep serving the old model
                log.exception("model load failed", extra={"stage_dir": str(self.stage_dir)})
                return self._model is not None
            finally:
                self._last_check = time.monotonic()

    def refresh_if_changed(self, force: bool = False) -> None:
        """Cheap periodic check; reload only when version.txt differs from memory."""
        if not force and time.monotonic() - self._last_check < self.CHECK_INTERVAL:
            return
        self._last_check = time.monotonic()
        disk = self._read_disk_version()
        current = self._model.version if self._model else None
        if disk is not None and disk != current:
            self.load()
