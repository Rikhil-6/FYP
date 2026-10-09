"""Evaluate a frozen ViT under progressive, reproducible temporal corruptions."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fyp_drift.data.clear100 import open_dataset
from fyp_drift.data.corruptions import (
    REALISTIC_BENCHMARK_SCENARIOS,
    SCENARIOS,
    drift_transform,
)
from fyp_drift.evaluation.classifier import evaluate
from fyp_drift.models.vision_transformer import TemporalViT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("baseline", "xai"), required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(REALISTIC_BENCHMARK_SCENARIOS),
    )
    parser.add_argument(
        "--long-horizon",
        action="store_true",
        help="Evaluate frozen checkpoints on test-source years 2012-2014",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TemporalViT(args.method, state["classifier.weight"].shape[0]).to(device)
    model.load_state_dict(state)
    dataset = open_dataset(args.data_root)
    year_column = dataset.metadata_fields.index("year")
    plan = (
        ((2012, "test_source", 1), (2013, "test_source", 2), (2014, "test_source", 3))
        if args.long_horizon
        else ((2012, "val", 1), (2013, "val", 2), (2014, "test", 3))
    )
    results = {
        "protocol": {
            "training_images_modified": False,
            "progression": {str(year): severity for year, _, severity in plan},
        },
        "scenarios": {},
    }
    for scenario in args.scenarios:
        results["scenarios"][scenario] = {}
        for year, split, severity in plan:
            subset = dataset.get_subset(
                split, transform=drift_transform(scenario, severity, args.seed)
            )
            year_id = year - 2002
            indices = torch.where(subset.metadata_array[:, year_column] == year_id)[
                0
            ].tolist()
            if args.limit:
                indices = indices[: args.limit]
            loader = DataLoader(
                torch.utils.data.Subset(subset, indices),
                batch_size=args.batch_size,
                num_workers=args.workers,
            )
            report, _ = evaluate(
                model, {"drift": loader}, year_column, device, splits=("drift",)
            )
            results["scenarios"][scenario][str(year)] = {
                "severity": 0 if scenario == "clean" else severity,
                "metrics": report["drift"]["all"],
                "timing": report["drift"]["_timing"],
            }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "drift_evaluation.json"
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                scenario: {
                    year: {
                        "macro_f1": row["metrics"]["macro_f1"],
                        "images_per_second": row["timing"]["images_per_second"],
                    }
                    for year, row in years.items()
                }
                for scenario, years in results["scenarios"].items()
            },
            indent=2,
        )
    )
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
