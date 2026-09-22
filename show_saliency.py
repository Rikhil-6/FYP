"""Export a held-out image and the XAI gate's 16x16 saliency overlay."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import MEAN, STD, make_loaders
from model import TemporalViT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.checkpoint.parent.parent / "metrics" / "saliency.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TemporalViT("xai", num_classes=checkpoint["classifier.weight"].shape[0]).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    loaders, _ = make_loaders(args.data_root, batch_size=1, limit_eval=1)
    image, label, metadata = next(iter(loaders["test"]))
    with torch.no_grad():
        logits, saliency, _ = model(image.to(device))
    rgb = image[0] * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    rgb = (rgb.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    base = Image.fromarray(rgb)
    heat = saliency[0].reshape(16, 16).cpu().numpy()
    heat = (255 * heat / max(heat.max(), 1e-8)).astype(np.uint8)
    heat = Image.fromarray(heat).resize(base.size, Image.Resampling.BILINEAR)
    red = Image.new("RGB", base.size, (255, 30, 30))
    overlay = Image.composite(red, base, heat.point(lambda value: int(value * 0.65)))
    canvas = Image.new("RGB", (448, 224))
    canvas.paste(base, (0, 0))
    canvas.paste(overlay, (224, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"saved={args.output} true={int(label[0])} predicted={int(logits.argmax(1)[0])}")


if __name__ == "__main__":
    main()
