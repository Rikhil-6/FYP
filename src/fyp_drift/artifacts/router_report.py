"""Export bounded, Git-friendly saliency panels for every drift condition."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from fyp_drift.data.clear100 import MEAN, STD, open_dataset
from fyp_drift.data.corruptions import (
    REALISTIC_BENCHMARK_SCENARIOS,
    SCENARIOS,
    drift_transform,
    scenario_components,
)
from fyp_drift.models.routing import apply_policy
from fyp_drift.models.siamese import ViTTwinRouter
from fyp_drift.models.vision_transformer import TemporalViT

SALIENCY_OVERLAY_STRENGTH = 0.28
SALIENCY_COLOR = (213, 94, 0)


def figure_font(size=14, bold=False):
    names = (
        ("timesbd.ttf", "DejaVuSerif-Bold.ttf")
        if bold
        else ("times.ttf", "DejaVuSerif.ttf")
    )
    roots = (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"))
    for root, name in zip(roots, names):
        candidate = root / name
        if candidate.is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def rgb(tensor):
    value = (
        tensor.cpu() * torch.tensor(STD)[:, None, None]
        + torch.tensor(MEAN)[:, None, None]
    )
    return Image.fromarray(
        (value.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )


def overlay(base, weights):
    heat = weights.reshape(16, 16).detach().cpu().numpy()
    heat = Image.fromarray(
        (255 * heat / max(heat.max(), 1e-8)).astype(np.uint8)
    ).resize(base.size)
    return Image.composite(
        Image.new("RGB", base.size, SALIENCY_COLOR),
        base,
        heat.point(lambda value: int(value * SALIENCY_OVERLAY_STRENGTH)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/reports/router")
    )
    parser.add_argument("--indices", type=int, nargs="+", default=(0, 777, 1777))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(REALISTIC_BENCHMARK_SCENARIOS),
    )
    parser.add_argument("--long-horizon", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = torch.load(args.base_checkpoint, map_location=device, weights_only=True)
    classes = base["classifier.weight"].shape[0]
    old, new = (
        TemporalViT("xai", classes).to(device),
        TemporalViT("xai", classes).to(device),
    )
    old.load_state_dict(base)
    new.load_state_dict(
        torch.load(
            args.run_dir / "models/best.pt", map_location=device, weights_only=True
        )
    )
    router = ViTTwinRouter(old, new).eval()
    policy = json.loads((args.run_dir / "metrics/router_policy.json").read_text())
    dataset = open_dataset()
    records = []
    years = (2012, 2013, 2014) if args.long_horizon else (2014,)
    split = "test_source" if args.long_horizon else "test"
    for scenario in args.scenarios:
        for severity, year in enumerate(years, 1):
            clean = dataset.get_subset(
                split, transform=drift_transform("clean", severity)
            )
            shifted = dataset.get_subset(
                split, transform=drift_transform(scenario, severity)
            )
            year_indices = torch.where(clean.metadata_array[:, 0] == year - 2002)[
                0
            ].tolist()
            folder = args.output / scenario / str(year)
            folder.mkdir(parents=True, exist_ok=True)
            for offset in args.indices:
                index = year_indices[offset % len(year_indices)]
                clean_image, label, _ = clean[index]
                image, _, _ = shifted[index]
                with torch.no_grad():
                    comparison = router.compare(image[None].to(device))
                    routed, flag, use_new, probability, features = apply_policy(
                        comparison, policy
                    )
                clean_rgb, shifted_rgb = rgb(clean_image), rgb(image)
                panels = [
                    clean_rgb,
                    shifted_rgb,
                    overlay(shifted_rgb, comparison["old_map"][0]),
                    overlay(shifted_rgb, comparison["new_map"][0]),
                ]
                canvas = Image.new("RGB", (896, 282), "white")
                names = (
                    "(a) Clean reference",
                    f"(b) {scenario.replace('_', ' ')}",
                    "(c) Historical evidence",
                    "(d) Updated evidence",
                )
                draw = ImageDraw.Draw(canvas)
                for position, (panel, name) in enumerate(zip(panels, names)):
                    canvas.paste(panel, (position * 224, 28))
                    draw.text(
                        (position * 224 + 6, 6),
                        name,
                        font=figure_font(14),
                        fill="black",
                    )
                footer = (
                    f"sample {index}   true {int(label)}   old {int(comparison['old_logits'].argmax(1)[0])}   "
                    f"new {int(comparison['new_logits'].argmax(1)[0])}   routed {int(routed[0])}   "
                    f"route probability {float(probability[0]):.3f}"
                )
                draw.text((6, 258), footer, font=figure_font(13), fill="black")
                path = folder / f"sample_{offset:04d}.png"
                canvas.save(path, optimize=True)
                records.append(
                    {
                        "scenario": scenario,
                        "year": year,
                        "severity": 0 if scenario == "clean" else severity,
                        "index": index,
                        "image": path.as_posix(),
                        "drift_components": list(
                            scenario_components(scenario, index, severity, seed=42)
                        ),
                        "true_class": int(label),
                        "old_class": int(comparison["old_logits"].argmax(1)[0]),
                        "new_class": int(comparison["new_logits"].argmax(1)[0]),
                        "routed_class": int(routed[0]),
                        "update_flag": bool(flag[0]),
                        "routed_to_new": bool(use_new[0]),
                        "route_probability": float(probability[0]),
                        **{
                            name: float(features[0, i])
                            for i, name in enumerate(policy["feature_names"])
                        },
                    }
                )
    (args.output / "samples.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
