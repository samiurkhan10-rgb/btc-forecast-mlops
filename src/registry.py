"""Minimal file-based model registry.

models/
  staging/<version>/      every trained candidate (pruned to the newest few)
  production/             the model the API serves (MODEL_STAGE=production)

Promotion copies a candidate next to the stage dir and swaps it in with renames,
so the API never reads a half-copied model.
"""

import logging
import math
import os
import shutil
from pathlib import Path

from src import config

log = logging.getLogger(__name__)


def passes_threshold(metrics: dict, threshold: float | None = None) -> bool:
    threshold = config.mape_threshold() if threshold is None else threshold
    mape = metrics.get("mape")
    return mape is not None and math.isfinite(mape) and mape <= threshold


def read_version(model_dir: Path) -> str | None:
    path = Path(model_dir) / config.VERSION_FILE
    return path.read_text().strip() if path.is_file() else None


def promote(candidate_dir: Path, models_root: Path | None = None, stage: str | None = None):
    candidate_dir = Path(candidate_dir)
    if read_version(candidate_dir) is None:
        raise FileNotFoundError(f"{candidate_dir} is not a complete model directory")
    root = Path(models_root or config.models_dir())
    stage = stage or config.model_stage()
    stage_dir = root / stage
    incoming = root / f".{stage}.incoming"
    old = root / f".{stage}.old"
    for leftover in (incoming, old):
        shutil.rmtree(leftover, ignore_errors=True)

    shutil.copytree(candidate_dir, incoming)
    if stage_dir.exists():
        os.replace(stage_dir, old)
    os.replace(incoming, stage_dir)
    shutil.rmtree(old, ignore_errors=True)
    log.info("model promoted", extra={"version": read_version(stage_dir), "stage": stage})
    return stage_dir


def prune_staging(models_root: Path | None = None, keep: int = 5) -> None:
    staging = Path(models_root or config.models_dir()) / "staging"
    if not staging.is_dir():
        return
    versions = sorted(p for p in staging.iterdir() if p.is_dir())
    for p in versions[:-keep]:
        shutil.rmtree(p, ignore_errors=True)


def register_if_passes(
    candidate_dir: Path, metrics: dict, threshold: float | None = None, **kwargs
) -> dict:
    threshold = config.mape_threshold() if threshold is None else threshold
    ok = passes_threshold(metrics, threshold)
    result = {
        "promoted": ok,
        "version": read_version(candidate_dir),
        "mape": metrics.get("mape"),
        "threshold": threshold,
    }
    if ok:
        promote(candidate_dir, **kwargs)
    else:
        log.warning("model rejected by quality gate", extra=result)
    return result
