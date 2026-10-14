"""Run a lightweight temporal replication on the Wild-Time Yearbook data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor
from transformers import Dinov2Model
from wild_time_data import load_dataset

from fyp_drift.data.clear100 import MEAN, STD
from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
from fyp_drift.evaluation.wilds_replication import (
    classification_scores,
    extract_features,
    fit_linear_probe,
    stratified_sample,
    train_architecture,
)
from fyp_drift.models.vision_transformer import BACKBONE


class WithYear(Dataset):
    """Add year metadata to the two-value Wild-Time dataset interface."""

    def __init__(self, dataset, year):
        self.dataset = dataset
        self.year = year
        self.labels = np.asarray(dataset.y, dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, torch.tensor([self.year])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare historical and later Wild-Time Yearbook periods."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/wild-time"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-start", type=int, default=1930)
    parser.add_argument("--train-end", type=int, default=1989)
    parser.add_argument("--id-start", type=int, default=1980)
    parser.add_argument("--id-end", type=int, default=1989)
    parser.add_argument("--ood-start", type=int, default=2004)
    parser.add_argument("--ood-end", type=int, default=2013)
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--permutation-limit", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--faithfulness-weight", type=float, default=0.1)
    return parser.parse_args(argv)


def transform():
    return Compose(
        [
            ToTensor(),
            Resize((224, 224), antialias=True),
            Lambda(lambda image: image.repeat(3, 1, 1)),
            Normalize(mean=MEAN, std=STD),
        ]
    )


def year_range(data_root, start, end, split):
    datasets = []
    for year in range(start, end + 1):
        dataset = load_dataset(
            "yearbook",
            time_step=year,
            split=split,
            transform=transform(),
            data_dir=str(data_root),
        )
        datasets.append(WithYear(dataset, year))
    combined = ConcatDataset(datasets)
    combined.labels = np.concatenate([dataset.labels for dataset in datasets])
    return combined


def sampled_loader(dataset, limit, seed, batch_size, workers):
    indices = stratified_sample(dataset.labels, limit, seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
    )


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = {
        "train": year_range(
            args.data_root, args.train_start, args.train_end, "train"
        ),
        "id": year_range(args.data_root, args.id_start, args.id_end, "test"),
        "ood": year_range(args.data_root, args.ood_start, args.ood_end, "test"),
    }
    loaders = {
        "train": sampled_loader(
            datasets["train"], args.train_limit, args.seed, args.batch_size, args.workers
        ),
        "id": sampled_loader(
            datasets["id"],
            args.eval_limit,
            args.seed + 1,
            args.batch_size,
            args.workers,
        ),
        "ood": sampled_loader(
            datasets["ood"],
            args.eval_limit,
            args.seed + 2,
            args.batch_size,
            args.workers,
        ),
    }

    backbone = Dinov2Model.from_pretrained(BACKBONE).to(device).eval()
    extracted = {
        name: extract_features(backbone, data_loader, device, name)
        for name, data_loader in loaders.items()
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
        "dataset": "Wild-Time Yearbook",
        "feature_extractor": BACKBONE,
        "seed": args.seed,
        "periods": {
            "training": [args.train_start, args.train_end],
            "in_distribution_test": [args.id_start, args.id_end],
            "later_test": [args.ood_start, args.ood_end],
        },
        "linear_probe": {
            "in_distribution": id_scores,
            "later_period": ood_scores,
            "macro_f1_change_percentage_points": 100
            * (ood_scores["macro_f1"] - id_scores["macro_f1"]),
        },
        "drift_detection": drift,
        "architecture_screen": architecture,
        "scope": (
            "This low-resolution binary task checks temporal reproducibility and "
            "the evaluation path. It does not replace a complex-image FMoW run."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "yearbook_replication.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
