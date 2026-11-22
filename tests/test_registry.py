import pytest

from src.registry import passes_threshold, promote, prune_staging, read_version, register_if_passes
from src.train_forecast import train


@pytest.mark.parametrize(
    "metrics, threshold, expected",
    [
        ({"mape": 3.0}, 10, True),
        ({"mape": 10.0}, 10, True),
        ({"mape": 12.5}, 10, False),
        ({"mape": float("nan")}, 10, False),
        ({}, 10, False),
    ],
)
def test_passes_threshold(metrics, threshold, expected):
    assert passes_threshold(metrics, threshold) is expected


def test_threshold_default_from_env(monkeypatch):
    monkeypatch.setenv("MAPE_THRESHOLD", "2")
    assert passes_threshold({"mape": 3.0}) is False


def test_promote_replaces_production(tmp_path, features_csv):
    root = tmp_path / "models"
    a = train(features_csv, root / "staging")
    promote(a["model_dir"], root, "production")
    assert read_version(root / "production") == a["version"]
    b = train(features_csv, root / "staging")
    promote(b["model_dir"], root, "production")
    assert read_version(root / "production") == b["version"]
    # no temp directories left behind
    assert sorted(p.name for p in root.iterdir()) == ["production", "staging"]


def test_promote_rejects_incomplete_dir(tmp_path):
    (tmp_path / "half").mkdir()
    with pytest.raises(FileNotFoundError):
        promote(tmp_path / "half", tmp_path, "production")


def test_register_if_passes_gate(tmp_path, features_csv):
    root = tmp_path / "models"
    res = train(features_csv, root / "staging")
    rejected = register_if_passes(res["model_dir"], {"mape": 99.0}, 10, models_root=root)
    assert rejected["promoted"] is False
    assert not (root / "production").exists()
    accepted = register_if_passes(res["model_dir"], res["metrics"], 1000, models_root=root)
    assert accepted["promoted"] is True
    assert read_version(root / "production") == res["version"]


def test_prune_staging_keeps_newest(tmp_path):
    staging = tmp_path / "staging"
    for v in ["20260101T000000Z", "20260102T000000Z", "20260103T000000Z"]:
        (staging / v).mkdir(parents=True)
    prune_staging(tmp_path, keep=2)
    assert sorted(p.name for p in staging.iterdir()) == ["20260102T000000Z", "20260103T000000Z"]
