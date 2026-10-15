"""Compare matched ViT heads across Camelyon17 hospitals."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import Dinov2Model

from fyp_drift.data.clear100 import MEAN, STD
from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
from fyp_drift.evaluation.wilds_replication import (
    classification_scores,
    extract_features,
    fit_linear_probe,
    train_architecture,
)
from fyp_drift.models.vision_transformer import BACKBONE


DATASET_NAME = "wltjr1007/Camelyon17-WILDS"


class ImageRows(Dataset):
    def __init__(self, rows, transform, domain):
        self.rows = rows
        self.transform = transform
        self.domain = domain
        self.labels = np.asarray([row[1] for row in rows], dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, label = self.rows[index]
        return self.transform(image), label, torch.tensor([self.domain])


def balanced_rows(stream, limit, seed):
    """Take an equal number of examples per class from a shuffled stream."""
    if limit % 2:
        raise ValueError("Camelyon sample limits must be even")
    quota = limit // 2
    rows = defaultdict(list)
    for example in stream.shuffle(seed=seed, buffer_size=20_000):
        label = int(example["label"])
        if len(rows[label]) < quota:
            rows[label].append((example["image"].convert("RGB").copy(), label))
        if len(rows[0]) == quota and len(rows[1]) == quota:
            break
    if len(rows[0]) != quota or len(rows[1]) != quota:
        raise ValueError("The stream did not contain enough examples from both classes")
    return [row for pair in zip(rows[0], rows[1]) for row in pair]


def balanced_source_rows(stream, train_limit, eval_limit, seed):
    """Create class-balanced source splits with disjoint patients."""
    quotas = {
        "train": train_limit // 2,
        "id": eval_limit // 2,
    }
    if train_limit % 2 or eval_limit % 2:
        raise ValueError("Camelyon sample limits must be even")
    rows = {name: defaultdict(list) for name in quotas}
    complete = False
    for example in stream.shuffle(seed=seed, buffer_size=20_000):
        patient = f"{example['center']}:{example['patient']}:{seed}"
        digest = hashlib.sha256(patient.encode()).digest()
        split = "id" if int.from_bytes(digest[:2]) % 5 == 0 else "train"
        label = int(example["label"])
        if len(rows[split][label]) < quotas[split]:
            image = example["image"].convert("RGB").copy()
            rows[split][label].append((image, label))
        complete = all(
            len(rows[name][label]) == quota
            for name, quota in quotas.items()
            for label in (0, 1)
        )
        if complete:
            break
    if not complete:
        raise ValueError("The source stream did not fill both patient-disjoint splits")
    return {
        name: [row for pair in zip(values[0], values[1]) for row in pair]
        for name, values in rows.items()
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Screen matched ViT heads on Camelyon17 hospital shift."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--permutation-limit", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--faithfulness-weight", type=float, default=0.1)
    return parser.parse_args(argv)


def image_transform():
    return Compose(
        [
            Resize((224, 224), antialias=True),
            ToTensor(),
            Normalize(mean=MEAN, std=STD),
        ]
    )


def main(argv=None):
    from datasets import load_dataset

    args = parse_args(argv)
    if args.train_limit < 100 or args.eval_limit < 100:
        raise ValueError("The train and evaluation limits must each be at least 100")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = load_dataset(DATASET_NAME, split="train", streaming=True)
    target = load_dataset(DATASET_NAME, split="validation", streaming=True)

    source_rows = balanced_source_rows(
        source, args.train_limit, args.eval_limit, args.seed
    )
    datasets = {
        "train": ImageRows(source_rows["train"], image_transform(), 0),
        "id": ImageRows(source_rows["id"], image_transform(), 0),
        "ood": ImageRows(
            balanced_rows(target, args.eval_limit, args.seed + 1),
            image_transform(),
            1,
        ),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
        )
        for name, dataset in datasets.items()
    }

    backbone = Dinov2Model.from_pretrained(BACKBONE).to(device).eval()
    extracted = {
        name: extract_features(backbone, loader, device, name)
        for name, loader in loaders.items()
    }
    del backbone
    if device == "cuda":
        torch.cuda.empty_cache()

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
    architecture = {
        method: train_architecture(method, 2, loaders, args, device)
        for method in ("baseline", "xai")
    }
    architecture["xai_minus_baseline_ood_macro_f1_points"] = 100 * (
        architecture["xai"]["out_of_distribution"]["macro_f1"]
        - architecture["baseline"]["out_of_distribution"]["macro_f1"]
    )

    payload = {
        "dataset": "Camelyon17-WILDS",
        "source": DATASET_NAME,
        "feature_extractor": BACKBONE,
        "seed": args.seed,
        "limits": {
            "training": len(datasets["train"]),
            "source_hospital_holdout": len(datasets["id"]),
            "validation_hospital": len(datasets["ood"]),
        },
        "linear_probe": {
            "in_distribution": id_scores,
            "out_of_distribution": ood_scores,
            "macro_f1_change_percentage_points": 100
            * (ood_scores["macro_f1"] - id_scores["macro_f1"]),
        },
        "drift_detection": drift,
        "architecture_screen": architecture,
        "scope": (
            "Camelyon17-WILDS tests acquisition shift across hospitals. The "
            "Hugging Face mirror groups source hospitals in its train split, "
            "so this screen uses a patient-disjoint balanced source holdout for ID "
            "evaluation and the validation hospital for OOD evaluation."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "camelyon17_replication.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
