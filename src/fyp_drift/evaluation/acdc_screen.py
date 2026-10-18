"""Screen historical and adapted ViT representations on paired ACDC views."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm.auto import tqdm

from fyp_drift.data.clear100 import MEAN, STD
from fyp_drift.evaluation.adaptation_evidence import (
    compare_checkpoints,
    compare_feature_archives,
)
from fyp_drift.models.vision_transformer import TemporalViT


class AcdcPairs(Dataset):
    def __init__(self, manifest: Path, limit=None, seed=42):
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        pair_ids = [row["pair_id"] for row in self.rows]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("ACDC manifest contains duplicate pair identifiers")
        if limit and limit < len(self.rows):
            by_condition = {}
            for row in self.rows:
                by_condition.setdefault(row["condition"], []).append(row)
            generator = random.Random(seed)
            conditions = sorted(by_condition)
            for rows in by_condition.values():
                generator.shuffle(rows)
            selected = []
            while len(selected) < limit:
                for condition in conditions:
                    if by_condition[condition]:
                        selected.append(by_condition[condition].pop())
                        if len(selected) == limit:
                            break
            self.rows = selected
        if len(self.rows) < 10:
            raise ValueError("The ACDC screen needs at least ten paired scenes")
        self.transform = Compose(
            [
                Resize((224, 224), antialias=True),
                ToTensor(),
                Normalize(mean=MEAN, std=STD),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["normal_image"]) as image:
            normal = self.transform(image.convert("RGB"))
        with Image.open(row["adverse_image"]) as image:
            adverse = self.transform(image.convert("RGB"))
        return normal, adverse, row["pair_id"], row["condition"]


def load_model(checkpoint: Path, method, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    classes = state["classifier.weight"].shape[0]
    model = TemporalViT(method=method, num_classes=classes)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def extract(model, loader, device, label):
    collected = {
        "normal_features": [],
        "adverse_features": [],
        "normal_probabilities": [],
        "adverse_probabilities": [],
        "normal_saliency": [],
        "adverse_saliency": [],
        "pair_ids": [],
        "conditions": [],
    }
    for normal, adverse, pair_ids, conditions in tqdm(loader, desc=label, unit="batch"):
        normal_features, normal_saliency, _, _ = model.encode(normal.to(device))
        adverse_features, adverse_saliency, _, _ = model.encode(adverse.to(device))
        collected["normal_features"].append(normal_features.cpu().numpy())
        collected["adverse_features"].append(adverse_features.cpu().numpy())
        collected["normal_probabilities"].append(
            model.classifier(normal_features).softmax(1).cpu().numpy()
        )
        collected["adverse_probabilities"].append(
            model.classifier(adverse_features).softmax(1).cpu().numpy()
        )
        collected["normal_saliency"].append(normal_saliency.cpu().numpy())
        collected["adverse_saliency"].append(adverse_saliency.cpu().numpy())
        collected["pair_ids"].extend(pair_ids)
        collected["conditions"].extend(conditions)
    return {
        name: np.concatenate(values, axis=0)
        if name not in ("pair_ids", "conditions")
        else np.asarray(values, dtype=str)
        for name, values in collected.items()
    }


def subset_archive(archive, indices):
    return {
        name: values[indices]
        for name, values in archive.items()
        if name not in ("pair_ids", "conditions")
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit paired ACDC representations for two XAI ViT checkpoints."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--historical-method", choices=("baseline", "xai"), default="xai"
    )
    parser.add_argument(
        "--adapted-method", choices=("baseline", "xai"), default="xai"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--permutation-limit", type=int, default=200)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--collapse-ratio-floor", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = AcdcPairs(args.manifest, args.limit, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    archives = {}
    for name, checkpoint, method in (
        ("historical", args.historical_checkpoint, args.historical_method),
        ("adapted", args.adapted_checkpoint, args.adapted_method),
    ):
        model = load_model(checkpoint, method, device)
        archives[name] = extract(model, loader, device, name)
        np.savez_compressed(args.output / f"{name}_features.npz", **archives[name])
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report = compare_feature_archives(
        archives["historical"],
        archives["adapted"],
        saliency_aligned=False,
        permutations=args.permutations,
        permutation_limit=args.permutation_limit,
        seed=args.seed,
        bootstrap_repetitions=args.bootstrap_repetitions,
        alpha=args.alpha,
        collapse_ratio_floor=args.collapse_ratio_floor,
    )
    report["parameter_update"] = compare_checkpoints(
        args.historical_checkpoint, args.adapted_checkpoint
    )
    conditions = sorted(set(archives["historical"]["conditions"].tolist()))
    report["by_condition"] = {}
    for condition in conditions:
        indices = np.flatnonzero(archives["historical"]["conditions"] == condition)
        report["by_condition"][condition] = compare_feature_archives(
            subset_archive(archives["historical"], indices),
            subset_archive(archives["adapted"], indices),
            saliency_aligned=False,
            permutations=args.permutations,
            permutation_limit=args.permutation_limit,
            seed=args.seed,
            bootstrap_repetitions=args.bootstrap_repetitions,
            alpha=args.alpha,
            collapse_ratio_floor=args.collapse_ratio_floor,
        )
    report["dataset"] = {
        "name": "ACDC",
        "pairs": len(dataset),
        "conditions": conditions,
        "scope": (
            "This is a paired representation screen for CLEAR-trained classifiers. "
            "It does not measure ACDC semantic-segmentation accuracy."
        ),
    }
    report["models"] = {
        "historical_method": args.historical_method,
        "adapted_method": args.adapted_method,
    }
    report_path = args.output / "adaptation_evidence.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
