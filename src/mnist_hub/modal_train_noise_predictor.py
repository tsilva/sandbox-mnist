from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import modal

from mnist_hub.train_noise_predictor import train_model


APP_NAME = "sandbox-mnist-noise-trainer"
VOLUME_NAME = "sandbox-mnist-training"
VOLUME_PATH = "/vol"


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "datasets>=3.4.1",
        "huggingface_hub>=0.31.0",
        "numpy>=2.2.4",
        "pillow>=11.1.0",
        "torch>=2.6.0",
        "torchvision>=0.21.0",
    )
    .add_local_python_source("mnist_hub")
)


def build_train_args(
    *,
    output_dir: str,
    dataset_repo: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    validation_fraction: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    seed: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_path=Path("/nonexistent-local-dataset"),
        dataset_repo=dataset_repo,
        output_dir=Path(output_dir),
        epochs=epochs,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        seed=seed,
        limit_train=None,
        limit_test=None,
        device="cuda",
    )


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 4,
    volumes={VOLUME_PATH: volume},
)
def train_remote(
    *,
    run_name: str,
    dataset_repo: str = "tsilva/mnist-gaussian-noisy",
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    validation_fraction: float = 0.1,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
    seed: int = 42,
) -> dict[str, object]:
    output_dir = f"{VOLUME_PATH}/runs/{run_name}"
    args = build_train_args(
        output_dir=output_dir,
        dataset_repo=dataset_repo,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        seed=seed,
    )
    metrics = train_model(args)
    volume.commit()
    return {
        "run_name": run_name,
        "output_dir": output_dir,
        "metrics": metrics,
    }


@app.local_entrypoint()
def main(
    run_name: str = "",
    dataset_repo: str = "tsilva/mnist-gaussian-noisy",
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    validation_fraction: float = 0.1,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
    seed: int = 42,
) -> None:
    effective_run_name = run_name or datetime.now(timezone.utc).strftime("conditioned-variance-%Y%m%d-%H%M%S")
    result = train_remote.remote(
        run_name=effective_run_name,
        dataset_repo=dataset_repo,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        seed=seed,
    )
    print(json.dumps(result, indent=2))
