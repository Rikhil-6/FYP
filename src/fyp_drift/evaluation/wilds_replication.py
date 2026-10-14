"""Screen for feature drift and performance loss on official WILDS splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from transformers import Dinov2Model
from wilds import get_dataset

from fyp_drift.data.clear100 import MEAN, STD
from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
from fyp_drift.models.vision_transformer import BACKBONE


DATASET_SPLITS = {
    "fmow": ("id_val", "val"),
    "iwildcam": ("id_val", "val"),
    "camelyon17": ("id_val", "val"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fit a frozen-DINOv2 linear probe and compare official WILDS "
            "in-distribution and out-of-distribution validation splits."
        )
    )
    parser.add_argument("--dataset", choices=DATASET_SPLITS, default="fmow")
    parser.add_argument("--data-root", type=Path, default=Path("data/wilds"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--permutation-limit", type=int, default=200)
    return parser.parse_args(argv)


def stratified_sample(labels, limit, seed):
    """Return a repeatable class-balanced sample without duplicating rows."""
    labels = np.asarray(labels).reshape(-1)
    buckets = {}
    for index, label in enumerate(labels):
        buckets.setdefault(int(label), []).append(index)

    generator = random.Random(seed)
    for indices in buckets.values():
        generator.shuffle(indices)

    selected = []
    target = min(limit, len(labels))
    while len(selected) < target:
        added = False
        for label in sorted(buckets):
            if buckets[label]:
                selected.append(buckets[label].pop())
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
    return sorted(selected)


def image_transform():
    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=MEAN, std=STD),
        ]
    )


def make_loader(dataset, split, limit, seed, batch_size, workers):
    subset = dataset.get_subset(split, transform=image_transform())
    labels = subset.y_array.cpu().numpy()
    indices = stratified_sample(labels, limit, seed)
    return DataLoader(
        Subset(subset, indices),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
    )


@torch.inference_mode()
def extract_features(model, data_loader, device, label):
    feature_batches = []
    label_batches = []
    for images, labels, _ in tqdm(data_loader, desc=label, unit="batch"):
        tokens = model(pixel_values=images.to(device)).last_hidden_state[:, 0]
        feature_batches.append(tokens.cpu().numpy())
        label_batches.append(labels.cpu().numpy())
    return np.concatenate(feature_batches), np.concatenate(label_batches)


def fit_linear_probe(features, labels, seed):
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )
    probe.fit(features, labels)
    return probe


def classification_scores(probe, features, labels):
    predictions = probe.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "samples": int(len(labels)),
    }


def main(argv=None):
    args = parse_args(argv)
    if args.train_limit < 100 or args.eval_limit < 100:
        raise ValueError("The train and evaluation limits must each be at least 100")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = get_dataset(
        dataset=args.dataset,
        root_dir=str(args.data_root),
        download=args.download,
    )
    id_split, ood_split = DATASET_SPLITS[args.dataset]
    required = {"train", id_split, ood_split}
    missing = required.difference(dataset.split_dict)
    if missing:
        raise ValueError(f"Dataset is missing required splits: {sorted(missing)}")

    model = Dinov2Model.from_pretrained(BACKBONE).to(device).eval()
    loaders = {
        "train": make_loader(
            dataset,
            "train",
            args.train_limit,
            args.seed,
            args.batch_size,
            args.workers,
        ),
        "id": make_loader(
            dataset,
            id_split,
            args.eval_limit,
            args.seed + 1,
            args.batch_size,
            args.workers,
        ),
        "ood": make_loader(
            dataset,
            ood_split,
            args.eval_limit,
            args.seed + 2,
            args.batch_size,
            args.workers,
        ),
    }
    extracted = {
        name: extract_features(model, data_loader, device, name)
        for name, data_loader in loaders.items()
    }
    probe = fit_linear_probe(*extracted["train"], args.seed)
    id_scores = classification_scores(probe, *extracted["id"])
    ood_scores = classification_scores(probe, *extracted["ood"])
    drift = distribution_shift_metrics(
        extracted["id"][0],
        extracted["ood"][0],
        permutations=args.permutations,
        permutation_limit=args.permutation_limit,
        seed=args.seed,
    )

    payload = {
        "dataset": args.dataset,
        "feature_extractor": BACKBONE,
        "device": device,
        "seed": args.seed,
        "splits": {
            "training": "train",
            "in_distribution": id_split,
            "out_of_distribution": ood_split,
        },
        "limits": {
            "training": args.train_limit,
            "evaluation_per_split": args.eval_limit,
        },
        "classification": {
            "in_distribution": id_scores,
            "out_of_distribution": ood_scores,
            "accuracy_change_percentage_points": 100
            * (ood_scores["accuracy"] - id_scores["accuracy"]),
            "macro_f1_change_percentage_points": 100
            * (ood_scores["macro_f1"] - id_scores["macro_f1"]),
        },
        "drift_detection": drift,
        "interpretation": (
            "MMD and domain-classifier AUC establish feature-distribution shift. "
            "The ID-to-OOD performance change measures whether that shift is harmful "
            "to this frozen-feature classifier."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{args.dataset}_replication.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
