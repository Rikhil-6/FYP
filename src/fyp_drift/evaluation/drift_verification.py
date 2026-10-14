"""Verify natural and controlled image drift with frozen DINOv2 features."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import Dinov2Model

from fyp_drift.data.clear100 import open_dataset
from fyp_drift.data.corruptions import (
    REALISTIC_BENCHMARK_SCENARIOS,
    drift_transform,
)
from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
from fyp_drift.models.vision_transformer import BACKBONE


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--permutation-limit", type=int, default=200)
    parser.add_argument("--reference-year", type=int, default=2009)
    parser.add_argument("--future-years", type=int, nargs="+", default=(2012, 2013, 2014))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=REALISTIC_BENCHMARK_SCENARIOS,
        default=list(REALISTIC_BENCHMARK_SCENARIOS[1:]),
    )
    return parser.parse_args(argv)


def stratified_indices(subset, year, limit, seed):
    year_id = year - 2002
    candidates = torch.where(subset.metadata_array[:, 0] == year_id)[0].tolist()
    by_class = {}
    for index in candidates:
        by_class.setdefault(int(subset.rows[index]["label"]), []).append(index)
    for label, indices in by_class.items():
        random.Random(seed + 1009 * label).shuffle(indices)

    selected = []
    while len(selected) < min(limit, len(candidates)):
        added = False
        for label in sorted(by_class):
            if by_class[label]:
                selected.append(by_class[label].pop())
                added = True
                if len(selected) == min(limit, len(candidates)):
                    break
        if not added:
            break
    return sorted(selected)


def loader(dataset, split, year, transform, indices, batch_size, workers):
    subset = dataset.get_subset(split, transform=transform)
    if indices is None:
        indices = stratified_indices(subset, year, len(subset), 42)
    return DataLoader(
        Subset(subset, indices),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
    )


@torch.inference_mode()
def extract_features(model, data_loader, device, label):
    rows = []
    for images, _, _ in tqdm(data_loader, desc=label, unit="batch"):
        tokens = model(pixel_values=images.to(device)).last_hidden_state[:, 0]
        rows.append(tokens.cpu().numpy())
    return np.concatenate(rows, axis=0)


def comparison_row(group, reference, target, first, second, args, paired):
    metrics = distribution_shift_metrics(
        first,
        second,
        permutations=args.permutations,
        permutation_limit=args.permutation_limit,
        seed=args.seed,
    )
    return {
        "group": group,
        "reference": reference,
        "target": target,
        "reference_samples": len(first),
        "target_samples": len(second),
        "paired": paired,
        **metrics,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    if args.limit < 10:
        raise ValueError("--limit must be at least 10")
    if args.permutations < 19:
        raise ValueError("--permutations must be at least 19")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Dinov2Model.from_pretrained(BACKBONE).to(device).eval()
    dataset = open_dataset(args.data_root)
    clean_transform = drift_transform("clean", 1, args.seed)

    reference_subset = dataset.get_subset("train_source", transform=clean_transform)
    reference_indices = stratified_indices(
        reference_subset, args.reference_year, args.limit, args.seed
    )
    reference_loader = loader(
        dataset,
        "train_source",
        args.reference_year,
        clean_transform,
        reference_indices,
        args.batch_size,
        args.workers,
    )
    reference_features = extract_features(
        model, reference_loader, device, f"clean {args.reference_year}"
    )

    rows = []
    for year in args.future_years:
        future_subset = dataset.get_subset("test_source", transform=clean_transform)
        indices = stratified_indices(future_subset, year, args.limit, args.seed)
        future_loader = loader(
            dataset,
            "test_source",
            year,
            clean_transform,
            indices,
            args.batch_size,
            args.workers,
        )
        features = extract_features(model, future_loader, device, f"clean {year}")
        rows.append(
            comparison_row(
                "natural_temporal",
                f"clean_{args.reference_year}",
                f"clean_{year}",
                reference_features,
                features,
                args,
                paired=False,
            )
        )

    evaluation_year = max(args.future_years)
    evaluation_subset = dataset.get_subset("test_source", transform=clean_transform)
    evaluation_indices = stratified_indices(
        evaluation_subset, evaluation_year, args.limit, args.seed
    )
    clean_loader = loader(
        dataset,
        "test_source",
        evaluation_year,
        clean_transform,
        evaluation_indices,
        args.batch_size,
        args.workers,
    )
    clean_features = extract_features(
        model, clean_loader, device, f"clean {evaluation_year} reference"
    )

    evaluation_labels = np.asarray(
        [int(evaluation_subset.rows[index]["label"]) for index in evaluation_indices]
    )
    first_indices, second_indices = train_test_split(
        np.arange(len(clean_features)),
        test_size=0.5,
        random_state=args.seed,
        stratify=evaluation_labels,
    )
    first_half = clean_features[first_indices]
    second_half = clean_features[second_indices]
    rows.append(
        comparison_row(
            "null_control",
            f"clean_{evaluation_year}_half_a",
            f"clean_{evaluation_year}_half_b",
            first_half,
            second_half,
            args,
            paired=False,
        )
    )

    severity = 3
    for scenario in args.scenarios:
        shifted_loader = loader(
            dataset,
            "test_source",
            evaluation_year,
            drift_transform(scenario, severity, args.seed),
            evaluation_indices,
            args.batch_size,
            args.workers,
        )
        shifted_features = extract_features(
            model, shifted_loader, device, f"{scenario} severity {severity}"
        )
        rows.append(
            comparison_row(
                "controlled_paired",
                f"clean_{evaluation_year}",
                f"{scenario}_severity_{severity}",
                clean_features,
                shifted_features,
                args,
                paired=True,
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "feature_extractor": BACKBONE,
            "reference_year": args.reference_year,
            "future_years": list(args.future_years),
            "controlled_year": evaluation_year,
            "controlled_severity": severity,
            "sample_limit": args.limit,
            "permutations": args.permutations,
            "permutation_samples_per_group": args.permutation_limit,
            "seed": args.seed,
            "interpretation": {
                "mmd_squared": "larger values indicate greater feature-distribution separation",
                "mmd_permutation_p": "small values reject exchangeability under the permutation test",
                "domain_classifier_roc_auc": "0.5 is chance and 1.0 is complete linear separability",
            },
        },
        "comparisons": rows,
    }
    json_path = args.output / "drift_verification.json"
    csv_path = args.output / "drift_verification.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    print(json.dumps(rows, indent=2))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
