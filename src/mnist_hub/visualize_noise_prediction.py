from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mnist_hub.train_noise_predictor import (
    NoisePredictor,
    image_to_tensor,
    load_split_dataset,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side example of noisy input, true noise, and predicted noise."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/smoke_noise_predictor/best_model.pt"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/mnist-gaussian-noisy"),
    )
    parser.add_argument("--dataset-repo", default="tsilva/mnist-gaussian-noisy")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/noise_prediction_sample.png"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def noise_to_heatmap(noise: np.ndarray, scale: float) -> Image.Image:
    norm = np.clip(noise / scale, -1.0, 1.0)
    heat = np.full((28, 28, 3), 245, dtype=np.uint8)

    neg = norm < 0
    pos = norm > 0

    if np.any(neg):
        values = (255 * (1.0 + norm[neg])).astype(np.uint8)
        heat[neg] = np.stack([values, values, np.full_like(values, 255)], axis=1)
    if np.any(pos):
        values = (255 * (1.0 - norm[pos])).astype(np.uint8)
        heat[pos] = np.stack([np.full_like(values, 255), values, values], axis=1)

    return Image.fromarray(heat, mode="RGB")


def load_model(checkpoint_path: Path, device: torch.device) -> NoisePredictor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = NoisePredictor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    dataset = load_split_dataset(args.dataset_path, args.dataset_repo)
    row = dataset[args.split][args.index]

    model = load_model(args.checkpoint, device)
    with torch.no_grad():
        input_tensor = image_to_tensor(row["image"]).unsqueeze(0).to(device)
        predicted_noise = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()

    true_noise = np.asarray(row["noise"], dtype=np.float32)
    scale = float(max(np.max(np.abs(true_noise)), np.max(np.abs(predicted_noise)), 1e-6))

    noisy_panel = row["image"].convert("L").convert("RGB")
    true_panel = noise_to_heatmap(true_noise, scale)
    pred_panel = noise_to_heatmap(predicted_noise, scale)

    scale_px = 8
    pad = 24
    title_h = 28
    panel_w = 28 * scale_px
    panel_h = 28 * scale_px
    width = pad * 4 + panel_w * 3
    height = pad * 2 + title_h + panel_h + 78

    canvas = Image.new("RGB", (width, height), (248, 246, 240))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Menlo.ttc", 18)
        meta_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Menlo.ttc", 15)
    except Exception:
        title_font = ImageFont.load_default()
        meta_font = title_font

    panels = [
        ("Noisy Image", noisy_panel.resize((panel_w, panel_h), Image.Resampling.NEAREST)),
        ("True Noise", true_panel.resize((panel_w, panel_h), Image.Resampling.NEAREST)),
        ("Predicted Noise", pred_panel.resize((panel_w, panel_h), Image.Resampling.NEAREST)),
    ]

    for idx, (title, image) in enumerate(panels):
        x = pad + idx * (panel_w + pad)
        y = pad + title_h
        draw.text((x, pad), title, fill=(25, 25, 25), font=title_font)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline=(210, 205, 198), width=2)

    mae = float(np.mean(np.abs(predicted_noise - true_noise)))
    meta = (
        f"split={args.split} index={args.index} label={row['label']} "
        f"source_index={row['source_index']} replica_index={row['replica_index']} "
        f"variance={row['noise_variance']:.4f}"
    )
    draw.text((pad, pad + title_h + panel_h + 18), meta, fill=(70, 70, 70), font=meta_font)
    draw.text(
        (pad, pad + title_h + panel_h + 40),
        f"shared color scale=±{scale:.4f}   prediction MAE={mae:.4f}",
        fill=(70, 70, 70),
        font=meta_font,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
