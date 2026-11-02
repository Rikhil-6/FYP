"""Measure how unlabelled CLEAR drift detection changes with sample size.

All images are embedded once with the frozen pretrained DINOv2 backbone.
Nested, unlabelled random 100/300/500-image samples make detector sensitivity
visible. Year metadata selects the comparison periods; class labels are never
read for sampling or by either detector.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Dinov2Model

from fyp_drift.data.clear100 import open_dataset
from fyp_drift.data.corruptions import drift_transform
from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
from fyp_drift.evaluation.drift_verification import (
    extract_features,
    loader,
)
from fyp_drift.models.vision_transformer import BACKBONE

DEFAULT_SIZES = (100, 300, 500)
CONTROLLED = ("motion_blur", "realistic_mix")
YEARS = (2009, 2012, 2013, 2014)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output already exists")
    if not args.sizes or any(size < 20 for size in args.sizes):
        parser.error("Each size must be at least 20")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("Sample sizes must be unique")
    if args.permutations < 19:
        parser.error("At least 19 permutations are required")
    return args


def unlabelled_indices(subset, year, count, seed):
    """Draw a fixed random order using only year metadata and image indices."""
    candidates = torch.where(subset.metadata_array[:, 0] == year - 2002)[0].tolist()
    if len(candidates) < count:
        raise ValueError(f"Year {year} has only {len(candidates)} images")
    np.random.default_rng(seed + year).shuffle(candidates)
    return candidates[:count]


def _select(features, maximum_indices, wanted_indices):
    positions = np.searchsorted(maximum_indices, wanted_indices)
    if not np.array_equal(maximum_indices[positions], wanted_indices):
        raise ValueError("Nested sample is outside the extracted maximum")
    return features[positions]


def _compare(first, second, *, paired, size, seed, permutations):
    return distribution_shift_metrics(
        first, second, paired=paired, seed=seed,
        permutations=permutations, permutation_limit=size,
    )


def main(argv=None):
    args = parse_args(argv)
    sizes = sorted(args.sizes)
    maximum = sizes[-1]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = Dinov2Model.from_pretrained(BACKBONE).to(device).eval()
    dataset = open_dataset(args.data_root)
    clean = drift_transform("clean", 1, args.seed)
    subset = dataset.get_subset("test_source", transform=clean)

    ordered = {}
    indices = {}
    features = {}
    for year in YEARS:
        count = maximum * (2 if year == 2014 else 1)
        ordered[year] = unlabelled_indices(subset, year, count, args.seed)
        indices[year] = sorted(ordered[year])
        data_loader = loader(
            dataset, "test_source", year, clean, indices[year],
            args.batch_size, 0,
        )
        features[year] = extract_features(
            backbone, data_loader, device, f"clean {year} max sample"
        )
        indices[year] = np.asarray(indices[year])

    adverse = {}
    controlled_indices = sorted(ordered[2014][:maximum])
    for scenario in CONTROLLED:
        data_loader = loader(
            dataset, "test_source", 2014,
            drift_transform(scenario, 3, args.seed), controlled_indices,
            args.batch_size, 0,
        )
        adverse[scenario] = extract_features(
            backbone, data_loader, device, f"{scenario} max sample"
        )
    controlled_indices = np.asarray(controlled_indices)

    comparisons = []
    for size in sizes:
        reference = _select(
            features[2009], indices[2009],
            sorted(ordered[2009][:size]),
        )
        for year in YEARS[1:]:
            future = _select(
                features[year], indices[year],
                sorted(ordered[year][:size]),
            )
            comparisons.append({
                "sample_size_per_group": size,
                "comparison": f"clean_2009_vs_clean_{year}",
                "paired": False,
                **_compare(reference, future, paired=False, size=size,
                           seed=args.seed, permutations=args.permutations),
            })
        first_indices = sorted(ordered[2014][:size])
        second_indices = sorted(ordered[2014][maximum:maximum + size])
        if set(first_indices) & set(second_indices):
            raise ValueError("Split-half null groups overlap")
        first = _select(features[2014], indices[2014], first_indices)
        second = _select(features[2014], indices[2014], second_indices)
        comparisons.append({
            "sample_size_per_group": size,
            "comparison": "clean_2014_disjoint_null",
            "paired": False,
            **_compare(first, second, paired=False, size=size,
                       seed=args.seed, permutations=args.permutations),
        })
        for scenario in CONTROLLED:
            shifted = _select(
                adverse[scenario], controlled_indices, first_indices
            )
            comparisons.append({
                "sample_size_per_group": size,
                "comparison": f"clean_2014_vs_{scenario}",
                "paired": True,
                **_compare(first, shifted, paired=True, size=size,
                           seed=args.seed, permutations=args.permutations),
            })
    report = {
        "dataset": "CLEAR-100 test_source",
        "extractor": BACKBONE,
        "input": "raw images embedded into frozen class-token features",
        "class_labels_used_by_detector": False,
        "class_labels_used_for_sampling": False,
        "sample_sizes_per_group": sizes,
        "permutations": args.permutations,
        "seed": args.seed,
        "interpretation": (
            "Non-detection at one sample size does not establish no drift. "
            "MMD and domain AUC describe distributional differences, not "
            "task harm. Unlabelled random sampling can change class mixture, "
            "so a retrospective class-balanced audit may help diagnose that "
            "confound but cannot be assumed in deployment."
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
