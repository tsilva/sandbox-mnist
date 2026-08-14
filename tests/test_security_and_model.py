from io import BytesIO
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from mnist_hub.build_datasets import _pil_image, variance_schedule
from mnist_hub.train_noise_predictor import NoisePredictor, build_model_input


def test_dependency_sources_are_registry_only() -> None:
    project = tomllib.load(open("pyproject.toml", "rb"))
    declared = list(project["project"]["dependencies"])
    for group in project["project"].get("optional-dependencies", {}).values():
        declared.extend(group)

    forbidden = (" @ ", "://", "file:", "git+")
    assert all(not any(marker in requirement for marker in forbidden) for requirement in declared)

    lock = tomllib.load(open("uv.lock", "rb"))
    for package in lock["package"]:
        source = package["source"]
        if package["name"] == "sandbox-mnist":
            assert "editable" in source
        else:
            assert source == {"registry": "https://pypi.org/simple"}


def test_valid_image_and_model_path_remain_functional() -> None:
    image = _pil_image(np.arange(28 * 28, dtype=np.uint8).reshape(28, 28))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    encoded.seek(0)

    with Image.open(encoded) as decoded:
        decoded.load()
        assert decoded.size == (28, 28)

    images = torch.rand(2, 1, 28, 28)
    labels = torch.tensor([0, 9])
    variances = torch.tensor([0.01, 0.10])
    model_input = build_model_input(images, labels, variances)
    output = NoisePredictor()(model_input)

    assert model_input.shape == (2, 12, 28, 28)
    assert output.shape == images.shape
    assert torch.isfinite(output).all()


def test_malformed_image_is_rejected() -> None:
    with pytest.raises((UnidentifiedImageError, OSError, SyntaxError)):
        with Image.open(BytesIO(b"\x89PNG\r\n\x1a\nmalformed")) as image:
            image.load()


@pytest.mark.parametrize(
    ("copies", "minimum", "maximum"),
    [(0, 0.01, 0.10), (2, -0.01, 0.10), (2, 0.20, 0.10)],
)
def test_invalid_variance_schedules_fail_closed(copies: int, minimum: float, maximum: float) -> None:
    with pytest.raises(ValueError):
        variance_schedule(copies, minimum, maximum)


def test_committed_checkpoint_loads_with_patched_torch() -> None:
    checkpoint_path = Path("runs/modal_variance_full_best_val_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = NoisePredictor()
    model.load_state_dict(checkpoint["model_state_dict"])

    assert checkpoint_path.is_file()
    assert isinstance(checkpoint.get("epoch"), int)
