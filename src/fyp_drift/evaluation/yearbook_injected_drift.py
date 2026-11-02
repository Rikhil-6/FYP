"""Paired drift screen for frozen Yearbook models and predeclared image shifts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
from fyp_drift.evaluation.metrics import classification_metrics
from fyp_drift.models.vision_transformer import TemporalViT
from fyp_drift.training.saliency_lora_pilot import build_model
from fyp_drift.training.yearbook_saliency_mask_pilot import make_loaders


def shifted(images, name, batch_index):
    if name == "clean":
        return images
    if name == "blur":
        return F.gaussian_blur(images, (17, 17), sigma=(3.0, 3.0)) if hasattr(F, "gaussian_blur") else _blur(images)
    if name == "center_occlusion":
        result = images.clone()
        side = result.shape[-1]
        start = side // 2 - side // 6
        result[:, :, start:start + side // 3, start:start + side // 3] = 0
        return result
    if name == "noise":
        generator = torch.Generator(device="cpu").manual_seed(42000 + batch_index)
        noise = torch.randn(images.shape, generator=generator).to(images.device)
        return images + 0.6 * noise
    raise ValueError(name)


def _blur(images):
    from torchvision.transforms.functional import gaussian_blur
    return gaussian_blur(images, 17, sigma=3.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/wild-time"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise ValueError("output already exists")
    args.output.mkdir(parents=True)
    config = json.loads((args.run / "config.json").read_text())
    settings = argparse.Namespace(**config)
    settings.data_root = args.data_root
    loader = make_loaders(settings)["future"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required")
    historical = TemporalViT("baseline", 2, pooling_mode="patch_only").to(device)
    historical.load_state_dict(torch.load(args.run / "historical.pt", map_location=device, weights_only=True))
    historical.eval()
    models = {"historical": historical}
    for name in ("none", "random", "saliency"):
        model = build_model(args.run / "historical.pt", device, settings.rank)
        model.load_state_dict(torch.load(args.run / f"{name}.pt", map_location=device, weights_only=True))
        models[name] = model.eval()
    payload = {"source_run": str(args.run), "seed": args.seed, "count": len(loader.dataset),
               "shift_parameters": {"blur": "Gaussian kernel 17 sigma 3 on 224 px", "center_occlusion": "central one-third square set to normalized mean", "noise": "Gaussian std 0.6 in normalized pixel units"},
               "scenarios": {}}
    clean_features = None
    for scenario in ("clean", "blur", "center_occlusion", "noise"):
        truth = []
        predictions = {name: [] for name in models}
        features = []
        years = []
        with torch.no_grad():
            for batch_index, (images, labels, batch_years) in enumerate(loader):
                images = shifted(images.to(device), scenario, batch_index)
                truth.extend(labels.tolist())
                years.extend(batch_years.squeeze(-1).tolist())
                tokens = historical.backbone(pixel_values=images).last_hidden_state[:, 1:]
                features.append(tokens.mean(1).cpu().numpy())
                predictions["historical"].extend(historical.classifier(tokens.mean(1)).argmax(1).cpu().tolist())
                for name in ("none", "random", "saliency"):
                    predictions[name].extend(models[name](images)[0].argmax(1).cpu().tolist())
        features = np.concatenate(features)
        result = {"models": {name: classification_metrics(truth, values, classes=2)
                             for name, values in predictions.items()}}
        if scenario == "clean":
            clean_features = features
            result["drift"] = None
            half = len(features) // 2
            random_order = np.random.default_rng(args.seed).permutation(len(features))
            payload["clean_split_half_control"] = distribution_shift_metrics(
                features[random_order[:half]], features[random_order[half:2 * half]],
                permutations=args.permutations, permutation_limit=half, seed=args.seed)
        else:
            result["drift"] = distribution_shift_metrics(
                clean_features, features, permutations=args.permutations,
                permutation_limit=200, seed=args.seed, paired=True)
            result["historical_accuracy_drop_points"] = 100 * (
                payload["scenarios"]["clean"]["models"]["historical"]["accuracy"]
                - result["models"]["historical"]["accuracy"])
        payload["scenarios"][scenario] = result
        with (args.output / f"{scenario}-predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("index", "year", "true_class", *models))
            writer.writerows((i, year, label, *(predictions[name][i] for name in models))
                             for i, (year, label) in enumerate(zip(years, truth)))
        print(json.dumps({"scenario": scenario, "result": result}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
