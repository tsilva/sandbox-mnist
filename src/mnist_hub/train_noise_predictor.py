from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image as PILImage
from torch import nn
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a model that predicts the additive noise residual from a noisy MNIST image."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/mnist-gaussian-noisy"),
        help="Local dataset path created with build_datasets.py.",
    )
    parser.add_argument(
        "--dataset-repo",
        default="tsilva/mnist-gaussian-noisy",
        help="Hugging Face dataset repo to use when --dataset-path does not exist.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/noise_predictor"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of the training split reserved for validation.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Number of epochs without validation improvement before stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss improvement required to reset patience.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split_dataset(dataset_path: Path, dataset_repo: str) -> DatasetDict:
    if dataset_path.exists():
        return load_from_disk(str(dataset_path))
    return load_dataset(dataset_repo)


def maybe_truncate(split: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(split):
        return split
    return split.select(range(limit))


def build_train_val_splits(split: Dataset, validation_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    split_dict = split.train_test_split(
        test_size=validation_fraction,
        seed=seed,
        stratify_by_column="label",
    )
    return split_dict["train"], split_dict["test"]


def image_to_tensor(image: PILImage.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, split: Dataset) -> None:
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.split[idx]
        return {
            "image": image_to_tensor(row["image"]),
            "noise": torch.tensor(row["noise"], dtype=torch.float32).unsqueeze(0),
            "variance": torch.tensor(row["noise_variance"], dtype=torch.float32),
        }


class NoisePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EpochMetrics:
    loss: float
    mae: float


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_mae = 0.0
    total_examples = 0
    criterion = nn.MSELoss()

    for batch in loader:
        images = batch["image"].to(device)
        target_noise = batch["noise"].to(device)
        batch_size = images.size(0)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            predicted_noise = model(images)
            loss = criterion(predicted_noise, target_noise)
            mae = torch.mean(torch.abs(predicted_noise - target_noise))
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size
        total_examples += batch_size

    return EpochMetrics(
        loss=total_loss / total_examples,
        mae=total_mae / total_examples,
    )


def estimate_baseline(loader: DataLoader, device: torch.device) -> EpochMetrics:
    total_loss = 0.0
    total_mae = 0.0
    total_examples = 0
    for batch in loader:
        target_noise = batch["noise"].to(device)
        zeros = torch.zeros_like(target_noise)
        batch_size = target_noise.size(0)
        mse = torch.mean((zeros - target_noise) ** 2)
        mae = torch.mean(torch.abs(target_noise))
        total_loss += mse.item() * batch_size
        total_mae += mae.item() * batch_size
        total_examples += batch_size
    return EpochMetrics(
        loss=total_loss / total_examples,
        mae=total_mae / total_examples,
    )


def build_loader(split: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        NoiseDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = load_split_dataset(args.dataset_path, args.dataset_repo)
    train_source = maybe_truncate(dataset["train"], args.limit_train)
    test_split = maybe_truncate(dataset["test"], args.limit_test)
    train_split, val_split = build_train_val_splits(
        train_source,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    train_loader = build_loader(train_split, args.batch_size, args.num_workers, shuffle=True)
    val_loader = build_loader(val_split, args.batch_size, args.num_workers, shuffle=False)
    test_loader = build_loader(test_split, args.batch_size, args.num_workers, shuffle=False)

    model = NoisePredictor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    baseline = estimate_baseline(test_loader, device)
    print(
        f"Baseline zero-predictor on test split: mse={baseline.loss:.6f} mae={baseline.mae:.6f}"
    )
    baseline_val = estimate_baseline(val_loader, device)
    print(
        f"Baseline zero-predictor on val split: mse={baseline_val.loss:.6f} mae={baseline_val.mae:.6f}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
        )
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            optimizer=None,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_mae": train_metrics.mae,
            "val_loss": val_metrics.loss,
            "val_mae": val_metrics.mae,
            "test_loss": test_metrics.loss,
            "test_mae": test_metrics.mae,
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics.loss:.6f} train_mae={train_metrics.mae:.6f} "
            f"val_loss={val_metrics.loss:.6f} val_mae={val_metrics.mae:.6f} "
            f"test_loss={test_metrics.loss:.6f} test_mae={test_metrics.mae:.6f}"
        )

        if val_metrics.loss < best_val_loss - args.early_stopping_min_delta:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics.loss,
                    "val_mae": val_metrics.mae,
                    "test_loss": test_metrics.loss,
                    "test_mae": test_metrics.mae,
                    "args": vars(args),
                },
                args.output_dir / "best_val_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} after "
                    f"{args.early_stopping_patience} epochs without validation improvement."
                )
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
            "best_epoch": best_epoch,
        },
        args.output_dir / "last_model.pt",
    )

    best_checkpoint = torch.load(
        args.output_dir / "best_val_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_val_metrics = run_epoch(
        model=model,
        loader=val_loader,
        device=device,
        optimizer=None,
    )
    best_test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
    )
    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "device": str(device),
                "train_examples": len(train_split),
                "val_examples": len(val_split),
                "test_examples": len(test_split),
                "baseline_val": asdict(baseline_val),
                "baseline_test": asdict(baseline),
                "best_epoch": best_epoch,
                "best_val": asdict(best_val_metrics),
                "best_test": asdict(best_test_metrics),
                "history": history,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
