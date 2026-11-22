"""Kubeflow Pipelines (KFP v2) definition of the BTC training workflow.

    ingest_data  ->  generate_features  ->  train_model
   (Dataset: raw)   (Dataset: features)   (Model + Metrics)

Each step is a *container component*: it runs `python -m src.cli <command>` inside the
training image built and pushed by CI (samiurk/btc-forecast-train). The image contains
src/ only - no kfp - because KFP's launcher injects itself at runtime. Data moves between
steps as typed KFP artifacts (`Dataset`, `Model`, `Metrics`), which KFP stores and tracks
(lineage, caching, UI).

Compile to YAML (needs only `pip install -r requirements-kfp.txt`, no cluster):
    python kubeflow/btc_kfp_pipeline.py --image-tag <git-sha>
"""

import argparse
import os
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

IMAGE_REPO = os.getenv("KFP_IMAGE_REPO", "docker.io/samiurk/btc-forecast-train")
CLI = ["python", "-m", "src.cli"]


def build_pipeline(image: str):
    """Create the components + pipeline for a given image.

    KFP evaluates a container component's ContainerSpec when the decorator runs, so the
    image must be known *before* the components are defined - hence this factory.
    """

    @dsl.container_component
    def ingest_data(raw_data: Output[Dataset]):
        """Pull daily BTC-USD OHLCV (yfinance -> CoinGecko -> Binance) into a CSV artifact."""
        return dsl.ContainerSpec(image=image, command=CLI, args=["fetch", "--out", raw_data.path])

    @dsl.container_component
    def generate_features(raw_data: Input[Dataset], features: Output[Dataset]):
        """Moving averages, volatility, lagged log returns and the next-day target."""
        return dsl.ContainerSpec(
            image=image,
            command=CLI,
            args=["features", "--raw", raw_data.path, "--out", features.path],
        )

    @dsl.container_component
    def train_model(features: Input[Dataset], model: Output[Model], metrics: Output[Metrics]):
        """XGBoost fit + recursive 7-day holdout evaluation. Writes the versioned model
        directory into the Model artifact and MAE/RMSE/MAPE JSON into the Metrics artifact."""
        return dsl.ContainerSpec(
            image=image,
            command=CLI,
            args=[
                "train",
                "--features",
                features.path,
                "--out-dir",
                model.path,
                "--metrics-out",
                metrics.path,
            ],
        )

    @dsl.pipeline(
        name="btc-forecast-training",
        description="Daily BTC-USD ingestion, feature generation and XGBoost 7-day training",
    )
    def btc_training_pipeline():
        ingest = ingest_data()
        ingest.set_caching_options(False)  # new market data every day: never reuse a fetch
        ingest.set_retry(num_retries=3, backoff_duration="60s", backoff_factor=2)

        feats = generate_features(raw_data=ingest.outputs["raw_data"])

        train = train_model(features=feats.outputs["features"])

        for step in (ingest, feats, train):
            step.set_cpu_request("250m").set_cpu_limit("1")
            step.set_memory_request("256Mi").set_memory_limit("1Gi")

    return btc_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-tag", default=os.getenv("KFP_IMAGE_TAG", "latest"))
    parser.add_argument("--output", default=str(Path(__file__).with_suffix(".yaml")))
    args = parser.parse_args()
    image = f"{IMAGE_REPO}:{args.image_tag}"
    compiler.Compiler().compile(build_pipeline(image), package_path=args.output)
    print(f"compiled {args.output} with image {image}")


if __name__ == "__main__":
    main()
