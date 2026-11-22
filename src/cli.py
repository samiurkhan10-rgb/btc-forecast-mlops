"""Command-line entry point used by the K8s training Job and the KFP container components.

python -m src.cli fetch    [--out RAW_CSV]
python -m src.cli features [--raw RAW_CSV] [--out FEATURES_CSV]
python -m src.cli train    [--features FEATURES_CSV] [--out-dir DIR] [--promote]
python -m src.cli all      (fetch -> features -> train -> promote if it passes the gate)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src import config
from src.logging_config import setup_logging

log = logging.getLogger("src.cli")


def _fetch(args) -> Path:
    from src.etl_fetch import run_fetch

    return run_fetch(Path(args.out) if args.out else None)


def _features(args) -> Path:
    from src.features import run_features

    raw = Path(args.raw) if args.raw else config.raw_path()
    out = Path(args.out) if args.out else config.features_path()
    return run_features(raw, out)


def _train(args) -> dict:
    from src.registry import prune_staging, register_if_passes
    from src.train_forecast import train

    features = Path(args.features) if args.features else config.features_path()
    result = train(features, Path(args.out_dir) if args.out_dir else None)
    if args.promote:
        result["registry"] = register_if_passes(
            Path(result["model_dir"]), result["metrics"], args.threshold
        )
        prune_staging()
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(result["metrics"], indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch")
    p.add_argument("--out")

    p = sub.add_parser("features")
    p.add_argument("--raw")
    p.add_argument("--out")

    for name in ("train", "all"):
        p = sub.add_parser(name)
        p.add_argument("--features")
        p.add_argument("--out-dir")
        p.add_argument("--promote", action="store_true", default=(name == "all"))
        p.add_argument("--threshold", type=float, default=None)
        p.add_argument("--metrics-out", help="also write metrics.json here (KFP)")
        if name == "all":
            p.add_argument("--raw")
            p.add_argument("--out")

    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            print(_fetch(args))
        elif args.command == "features":
            print(_features(args))
        elif args.command == "train":
            print(json.dumps(_summary(_train(args))))
        else:  # all
            args.raw = str(_fetch(args))
            args.out = None
            args.features = str(_features(args))
            print(json.dumps(_summary(_train(args))))
    except Exception:  # noqa: BLE001 - log any failure as JSON and exit non-zero
        log.exception("command failed", extra={"command": args.command})
        return 1
    return 0


def _summary(result: dict) -> dict:
    m = result["metrics"]
    return {
        "version": result["version"],
        "model_dir": result["model_dir"],
        "mae": round(m["mae"], 2),
        "rmse": round(m["rmse"], 2),
        "mape": round(m["mape"], 3),
        "registry": result.get("registry"),
    }


if __name__ == "__main__":
    sys.exit(main())
