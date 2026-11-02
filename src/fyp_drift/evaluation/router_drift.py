"""Reevaluate saved historical and updated ViTs without retraining them."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from fyp_drift.data.clear100 import balanced_indices, open_dataset
from fyp_drift.data.corruptions import SCENARIOS, drift_transform, scenario_components
from fyp_drift.training.temporal_router import (
    build_router,
    evaluate_stream,
    seed_everything,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate an existing router checkpoint under named drift conditions."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=("realistic_mix",)
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args(argv)


def _write_decisions(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    policy = json.loads(
        (args.run_dir / "metrics" / "router_policy.json").read_text(encoding="utf-8")
    )
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    profile = config.get("adaptation_profile", "standard")
    router, _, _, _ = build_router(
        Path(config["base_checkpoint"]), device, profile,
        initialize_saliency_from_uniform=config.get(
            "initialize_saliency_from_uniform", False
        ),
    )
    updated_state = torch.load(
        args.run_dir / "models" / "best.pt", map_location=device, weights_only=True
    )
    router.new.load_state_dict(updated_state)

    data_root = args.data_root or Path(config.get("data_root", "data"))
    dataset = open_dataset(data_root)
    year_column = dataset.metadata_fields.index("year")
    test_split = "test_source" if config.get("long_horizon") else "test"
    test_years = tuple(config.get("test_years", (2014,)))
    limit_test = config.get("limit_test")
    batch_size = args.batch_size or int(config.get("batch_size", 8))

    results = {
        "source_run": str(args.run_dir),
        "checkpoint": "models/best.pt",
        "policy": "metrics/router_policy.json",
        "training_performed": False,
        "scenarios": {},
    }
    all_decisions = []
    for scenario in args.scenarios:
        results["scenarios"][scenario] = {}
        for severity, year in enumerate(test_years, 1):
            subset = dataset.get_subset(
                test_split,
                transform=drift_transform(scenario, severity, seed),
            )
            indices = torch.where(
                subset.metadata_array[:, year_column] == year - 2002
            )[0].tolist()
            if limit_test:
                indices = balanced_indices(
                    subset.rows, indices, min(int(limit_test), len(indices)), seed
                )
            loader = DataLoader(
                Subset(subset, indices),
                batch_size=batch_size,
                num_workers=args.workers,
            )
            report, decisions = evaluate_stream(
                router, loader, policy, device, year_column
            )
            for dataset_index, decision in zip(indices, decisions):
                all_decisions.append(
                    {
                        "scenario": scenario,
                        "severity": 0 if scenario == "clean" else severity,
                        "dataset_index": dataset_index,
                        "drift_components": "+".join(
                            scenario_components(scenario, dataset_index, severity, seed)
                        ),
                        **decision,
                    }
                )
            results["scenarios"][scenario][str(year)] = {
                "severity": 0 if scenario == "clean" else severity,
                "results": report,
            }
            summary = report["all"]
            print(
                json.dumps(
                    {
                        "scenario": scenario,
                        "year": year,
                        "historical_macro_f1": summary["old"]["macro_f1"],
                        "updated_macro_f1": summary["new"]["macro_f1"],
                        "routed_macro_f1": summary["routed"]["macro_f1"],
                        "helpful_routes": summary["helpful_routes"],
                        "harmful_routes": summary["harmful_routes"],
                    }
                ),
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "router_drift_evaluation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    _write_decisions(args.output / "routing_decisions.csv", all_decisions)


if __name__ == "__main__":
    main()
