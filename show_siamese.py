"""Visualize the two saliency maps and the transparent routing decision."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import MEAN, STD, open_dataset, year_loader
from model import TemporalViT
from siamese import ViTTwinRouter


SALIENCY_OVERLAY_STRENGTH = 0.35


def overlay(base, weights):
    heat = weights.reshape(16, 16).detach().cpu().numpy()
    heat = Image.fromarray((255 * heat / max(heat.max(), 1e-8)).astype(np.uint8))
    heat = heat.resize(base.size, Image.Resampling.BILINEAR)
    return Image.composite(
        Image.new("RGB", base.size, (255, 25, 25)),
        base,
        heat.point(lambda value: int(value * SALIENCY_OVERLAY_STRENGTH)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.run_dir / "metrics" / "siamese_maps.png"
    if args.index < 0:
        raise ValueError("--index must be nonnegative")
    thresholds = json.loads((args.run_dir / "metrics" / "thresholds.json").read_text())
    thresholds = {int(key) if key != "global" else key: value for key, value in thresholds.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = torch.load(args.base_checkpoint, map_location=device, weights_only=True)
    num_classes = base["classifier.weight"].shape[0]
    old = TemporalViT("xai", num_classes=num_classes).to(device)
    new = TemporalViT("xai", num_classes=num_classes).to(device)
    old.load_state_dict(base)
    new.load_state_dict(torch.load(args.run_dir / "models" / "best.pt", map_location=device, weights_only=True))
    router = ViTTwinRouter(old, new).eval()
    dataset = open_dataset(args.data_root)
    loader, _ = year_loader(dataset, "test", (2014,), batch_size=1, limit=args.index + 1)
    if len(loader.dataset) <= args.index:
        raise IndexError("--index exceeds the future test set")
    for image, label, _ in loader:
        pass
    with torch.no_grad():
        comparison = router.compare(image.to(device))
        prediction, needs_update, use_new = router.route(comparison, thresholds)
    rgb = image[0] * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    base = Image.fromarray((rgb.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    canvas = Image.new("RGB", (672, 224))
    canvas.paste(base, (0, 0))
    canvas.paste(overlay(base, comparison["old_map"][0]), (224, 0))
    canvas.paste(overlay(base, comparison["new_map"][0]), (448, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    old_class = int(comparison["old_logits"].argmax(1)[0])
    print(json.dumps({"image": str(args.output), "panels": ["image", "old map", "new map"],
                      "true_class": int(label[0]), "old_class": old_class,
                      "new_class": int(comparison["new_logits"].argmax(1)[0]),
                      "saliency_js": float(comparison["saliency_gap"][0]),
                      "class_threshold": thresholds.get(old_class, thresholds["global"]),
                      "old_evidence": float(comparison["old_evidence"][0]),
                      "new_evidence": float(comparison["new_evidence"][0]),
                      "needs_update": bool(needs_update[0]),
                      "routed_to_new": bool(use_new[0]),
                      "routed_class": int(prediction[0])}, indent=2))


if __name__ == "__main__":
    main()
