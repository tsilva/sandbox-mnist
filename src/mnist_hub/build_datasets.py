from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Iterable

import numpy as np
from datasets import Array2D, ClassLabel, Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image as PILImage
from torchvision.datasets import MNIST


LABEL_FEATURE = ClassLabel(names=[str(i) for i in range(10)])
BASE_FEATURES = Features(
    {
        "image": Image(),
        "label": LABEL_FEATURE,
        "source_index": Value("int32"),
    }
)
NOISY_FEATURES = Features(
    {
        "image": Image(),
        "noise": Array2D(shape=(28, 28), dtype="float32"),
        "raw_image": Image(),
        "label": LABEL_FEATURE,
        "source_index": Value("int32"),
        "replica_index": Value("int16"),
        "noise_variance": Value("float32"),
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MNIST and Gaussian-noisy MNIST datasets and optionally push them to the Hugging Face Hub."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--namespace", help="Hugging Face username or org. Inferred from token when omitted.")
    parser.add_argument("--base-repo", default="mnist")
    parser.add_argument("--noisy-repo", default="mnist-gaussian-noisy")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--copies-per-example", type=int, default=5)
    parser.add_argument(
        "--variance-min",
        type=float,
        default=0.01,
        help="Minimum Gaussian variance on normalized [0, 1] image values.",
    )
    parser.add_argument(
        "--variance-max",
        type=float,
        default=0.10,
        help="Maximum Gaussian variance on normalized [0, 1] image values.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    return parser.parse_args()


def _pil_image(array: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(array, mode="L")


def load_mnist_split(data_dir: Path, train: bool) -> MNIST:
    return MNIST(root=str(data_dir), train=train, download=True)


def build_base_split(data_dir: Path, train: bool) -> Dataset:
    raw = load_mnist_split(data_dir, train=train)
    indices = split_indices(raw.targets.tolist(), train=train)
    images = [_pil_image(raw.data[idx].numpy()) for idx in indices]
    labels = [int(raw.targets[idx]) for idx in indices]
    source_index = indices
    return Dataset.from_dict(
        {
            "image": images,
            "label": labels,
            "source_index": source_index,
        },
        features=BASE_FEATURES,
    )


def build_base_dataset(data_dir: Path) -> DatasetDict:
    return DatasetDict(
        {
            "train": build_base_split(data_dir, train=True),
            "test": build_base_split(data_dir, train=False),
        }
    )


def split_indices(targets: list[int], train: bool) -> list[int]:
    if train:
        return list(range(len(targets)))

    label_to_indices: dict[int, list[int]] = {label: [] for label in range(10)}
    for idx, label in enumerate(int(target) for target in targets):
        label_to_indices[label].append(idx)

    per_class = min(len(indices) for indices in label_to_indices.values())
    balanced_indices = []
    for label in range(10):
        balanced_indices.extend(label_to_indices[label][:per_class])
    return balanced_indices


def variance_schedule(copies_per_example: int, variance_min: float, variance_max: float) -> np.ndarray:
    if copies_per_example < 1:
        raise ValueError("--copies-per-example must be at least 1")
    if variance_min < 0 or variance_max < 0:
        raise ValueError("Noise variances must be non-negative")
    if variance_min > variance_max:
        raise ValueError("--variance-min cannot be greater than --variance-max")
    return np.linspace(variance_min, variance_max, copies_per_example, dtype=np.float32)


def noisy_examples(
    *,
    data_dir: Path,
    train: bool,
    variances: np.ndarray,
    seed: int,
) -> Iterable[dict]:
    raw = load_mnist_split(data_dir, train=train)
    rng = np.random.default_rng(seed + (0 if train else 1))
    for idx in split_indices(raw.targets.tolist(), train=train):
        base = raw.data[idx].numpy().astype(np.float32) / 255.0
        label = int(raw.targets[idx])
        for replica_index, variance in enumerate(variances):
            std = math.sqrt(float(variance))
            sampled_noise = rng.normal(loc=0.0, scale=std, size=base.shape).astype(np.float32)
            noisy = np.clip(base + sampled_noise, 0.0, 1.0)
            yield {
                "image": _pil_image((noisy * 255.0).round().astype(np.uint8)),
                "noise": sampled_noise,
                "raw_image": _pil_image(raw.data[idx].numpy()),
                "label": label,
                "source_index": idx,
                "replica_index": replica_index,
                "noise_variance": float(variance),
            }


def build_noisy_split(data_dir: Path, train: bool, variances: np.ndarray, seed: int) -> Dataset:
    return Dataset.from_generator(
        noisy_examples,
        gen_kwargs={
            "data_dir": data_dir,
            "train": train,
            "variances": variances,
            "seed": seed,
        },
        features=NOISY_FEATURES,
    )


def build_noisy_dataset(data_dir: Path, variances: np.ndarray, seed: int) -> DatasetDict:
    return DatasetDict(
        {
            "train": build_noisy_split(data_dir, train=True, variances=variances, seed=seed),
            "test": build_noisy_split(data_dir, train=False, variances=variances, seed=seed),
        }
    )


def save_dataset(dataset: DatasetDict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    dataset.save_to_disk(str(path))


def resolve_namespace(token: str | None, explicit_namespace: str | None) -> str | None:
    if explicit_namespace:
        return explicit_namespace
    if not token:
        return None
    api = HfApi(token=token)
    return api.whoami()["name"]


def preserve_front_matter(existing_readme: str, body: str) -> str:
    if not existing_readme.startswith("---\n"):
        return body.strip() + "\n"
    closing = existing_readme.find("\n---\n", 4)
    if closing == -1:
        return body.strip() + "\n"
    front_matter = existing_readme[: closing + len("\n---\n")]
    return front_matter + "\n" + body.strip() + "\n"


def format_variances(variances: np.ndarray) -> str:
    return ", ".join(f"{float(variance):.4f}" for variance in variances)


def base_card_body(repo_id: str) -> str:
    return dedent(
        f"""
        # {repo_id}

        ## Dataset Summary

        This dataset is a Hugging Face packaged version of the classic MNIST handwritten digits benchmark. It contains grayscale 28x28 images of digits `0` through `9`, split into the standard `train` and `test` partitions.

        Each row includes:

        - `image`: a 28x28 grayscale digit image
        - `label`: the digit class from `0` to `9`
        - `source_index`: the original position of the example inside the source MNIST split

        ## Splits

        - `train`: 60,000 examples
        - `test`: 8,920 examples, balanced to `892` examples per class

        ## Source

        The underlying data comes from the original MNIST release maintained by Yann LeCun and collaborators and downloaded here through `torchvision.datasets.MNIST`.

        - MNIST homepage: http://yann.lecun.com/exdb/mnist/
        - TorchVision dataset docs: https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html

        ## Intended Use

        This dataset is suitable for standard handwritten digit classification baselines, representation learning experiments, and as a clean reference set for synthetic corruption studies.

        ## Load Example

        ```python
        from datasets import load_dataset

        ds = load_dataset("{repo_id}")
        print(ds["train"][0])
        ```
        """
    ).strip()


def noisy_card_body(
    repo_id: str,
    *,
    copies_per_example: int,
    variances: np.ndarray,
    seed: int,
) -> str:
    return dedent(
        f"""
        # {repo_id}

        ## Dataset Summary

        This dataset expands MNIST by creating multiple Gaussian-noisy variants of each original example. Each row is structured for direct supervised training: the input is a noisy image and the target is the original sampled Gaussian noise map, with the clean image kept as a reference column.

        Noise is sampled from a zero-mean normal distribution on normalized pixel values in `[0, 1]`, added to the clean image, clipped back to `[0, 1]`, and converted to 8-bit grayscale. The `noise` column stores the original sampled Gaussian draw before clipping.

        ## Columns

        - `image`: the noisy 28x28 grayscale input image used as the model source
        - `noise`: the 28x28 float Gaussian noise sample in normalized pixel space
        - `raw_image`: the clean 28x28 grayscale reference image
        - `label`: the original digit class from `0` to `9`
        - `source_index`: the original example index inside the source MNIST split
        - `replica_index`: which noisy replica this row corresponds to for the clean source image
        - `noise_variance`: the Gaussian variance used to sample the stored noise map

        ## Splits

        - `train`: 300,000 image pairs
        - `test`: 44,600 image pairs, balanced to `4,460` pairs per class

        ## Noise Configuration

        - Source dataset: MNIST
        - Noisy counterparts per source example: `{copies_per_example}`
        - Variances: `{format_variances(variances)}`
        - Random seed: `{seed}`
        - Test balancing: exact class balance via downsampling the MNIST test split to the minimum class count

        ## Intended Use

        This dataset is intended for experiments where each training row should already contain a noisy source image and the original noise sample used to corrupt it. It is suited for noise prediction and generative or iterative denoising setups that operate directly on sampled noise fields.

        ## Load Example

        ```python
        from datasets import load_dataset

        ds = load_dataset("{repo_id}")
        sample = ds["train"][0]
        print(sample["image"])
        print(sample["noise"][0][0])
        print(sample["raw_image"])
        print(sample["noise_variance"])
        ```
        """
    ).strip()


def publish_dataset_card(repo_id: str, token: str, body: str) -> None:
    api = HfApi(token=token)
    readme_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="README.md", token=token)
    existing_readme = readme_path and Path(readme_path).read_text()
    full_readme = preserve_front_matter(existing_readme, body)
    api.upload_file(
        path_or_fileobj=full_readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


def push_dataset(
    dataset: DatasetDict,
    repo_id: str,
    token: str,
    private: bool,
    card_body: str,
) -> None:
    HfApi(token=token).create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    dataset.push_to_hub(repo_id, token=token, private=private)
    publish_dataset_card(repo_id, token, card_body)


def main() -> None:
    args = parse_args()
    variances = variance_schedule(args.copies_per_example, args.variance_min, args.variance_max)

    print("Building base MNIST dataset...")
    base_dataset = build_base_dataset(args.data_dir)
    base_path = args.output_dir / "mnist"
    save_dataset(base_dataset, base_path)
    print(f"Saved base dataset to {base_path}")

    print("Building noisy MNIST dataset...")
    noisy_dataset = build_noisy_dataset(args.data_dir, variances, args.seed)
    noisy_path = args.output_dir / "mnist-gaussian-noisy"
    save_dataset(noisy_dataset, noisy_path)
    print(f"Saved noisy dataset to {noisy_path}")

    namespace = resolve_namespace(args.token, args.namespace)
    if args.skip_push or not args.token or not namespace:
        print("Skipping hub upload.")
        if not args.token:
            print("Reason: no Hugging Face token was provided. Set HF_TOKEN or pass --token.")
        elif not namespace:
            print("Reason: namespace could not be resolved. Pass --namespace explicitly.")
        return

    base_repo_id = f"{namespace}/{args.base_repo}"
    noisy_repo_id = f"{namespace}/{args.noisy_repo}"

    print(f"Pushing base dataset to {base_repo_id}...")
    push_dataset(
        base_dataset,
        base_repo_id,
        args.token,
        args.private,
        base_card_body(base_repo_id),
    )
    print(f"Pushing noisy dataset to {noisy_repo_id}...")
    push_dataset(
        noisy_dataset,
        noisy_repo_id,
        args.token,
        args.private,
        noisy_card_body(
            noisy_repo_id,
            copies_per_example=args.copies_per_example,
            variances=variances,
            seed=args.seed,
        ),
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
