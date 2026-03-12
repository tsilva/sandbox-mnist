from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mnist_hub.train_noise_predictor import NoisePredictor, build_model_input, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteratively denoise random noise with a label-conditioned MNIST noise predictor."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/full_conditioned_noise_predictor/best_val_model.pt"),
    )
    parser.add_argument("--label", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance", type=float, default=0.10)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/conditioned_iterative_sampling.png"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def render_grid(images: list[tuple[str, np.ndarray]], output: Path, meta: str) -> None:
    scale_px = 8
    pad = 24
    title_h = 28
    panel_w = 28 * scale_px
    panel_h = 28 * scale_px
    width = pad * (len(images) + 1) + panel_w * len(images)
    height = pad * 2 + title_h + panel_h + 52

    canvas = Image.new("RGB", (width, height), (248, 246, 240))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Menlo.ttc", 18)
        meta_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Menlo.ttc", 15)
    except Exception:
        title_font = ImageFont.load_default()
        meta_font = title_font

    for idx, (title, array) in enumerate(images):
        x = pad + idx * (panel_w + pad)
        y = pad + title_h
        image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L").convert("RGB")
        image = image.resize((panel_w, panel_h), Image.Resampling.NEAREST)
        draw.text((x, pad), title, fill=(25, 25, 25), font=title_font)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline=(210, 205, 198), width=2)

    draw.text((pad, pad + title_h + panel_h + 18), meta, fill=(70, 70, 70), font=meta_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NoisePredictor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    sampled_noise = rng.normal(
        loc=0.0,
        scale=math.sqrt(args.variance),
        size=(28, 28),
    ).astype(np.float32)
    current = np.clip(0.5 + sampled_noise, 0.0, 1.0)

    targets = sorted(set(args.steps))
    renders: list[tuple[str, np.ndarray]] = [("Step 0", current.copy())]

    with torch.no_grad():
        for step in range(1, targets[-1] + 1):
            image_tensor = torch.from_numpy(current).unsqueeze(0).unsqueeze(0).to(device)
            label_tensor = torch.tensor([args.label], dtype=torch.long, device=device)
            variance_tensor = torch.tensor([args.variance], dtype=torch.float32, device=device)
            model_input = build_model_input(image_tensor, label_tensor, variance_tensor)
            predicted_noise = model(model_input).squeeze(0).squeeze(0).cpu().numpy()
            current = np.clip(current - predicted_noise, 0.0, 1.0)
            if step in targets:
                renders.append((f"Step {step}", current.copy()))

    meta = (
        f"label={args.label} seed={args.seed} variance={args.variance:.4f} "
        f"checkpoint_epoch={checkpoint.get('epoch', '?')}"
    )
    render_grid(renders, args.output, meta)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
