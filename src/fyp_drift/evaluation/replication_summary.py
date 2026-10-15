"""Summarise matched architecture screens without pooling different datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarise external replication runs.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load_row(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    screen = result["architecture_screen"]
    baseline = screen["baseline"]
    xai = screen["xai"]
    drift = result["drift_detection"]
    return {
        "dataset": result["dataset"],
        "seed": result["seed"],
        "mmd_squared": drift["mmd_squared"],
        "mmd_p": drift["mmd_permutation_p"],
        "domain_auc": drift["domain_classifier_roc_auc"],
        "baseline_id_macro_f1": baseline["in_distribution"]["macro_f1"],
        "baseline_ood_macro_f1": baseline["out_of_distribution"]["macro_f1"],
        "xai_id_macro_f1": xai["in_distribution"]["macro_f1"],
        "xai_ood_macro_f1": xai["out_of_distribution"]["macro_f1"],
        "xai_minus_baseline_ood_points": screen[
            "xai_minus_baseline_ood_macro_f1_points"
        ],
    }


def summarise(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    summaries = []
    for dataset, dataset_rows in sorted(grouped.items()):
        differences = [row["xai_minus_baseline_ood_points"] for row in dataset_rows]
        summaries.append(
            {
                "dataset": dataset,
                "runs": len(dataset_rows),
                "baseline_ood_macro_f1_mean": fmean(
                    row["baseline_ood_macro_f1"] for row in dataset_rows
                ),
                "xai_ood_macro_f1_mean": fmean(
                    row["xai_ood_macro_f1"] for row in dataset_rows
                ),
                "xai_minus_baseline_ood_points_mean": fmean(differences),
                "xai_minus_baseline_ood_points_std": (
                    stdev(differences) if len(differences) > 1 else None
                ),
            }
        )
    return summaries


def main(argv=None):
    args = parse_args(argv)
    rows = [load_row(path) for path in args.inputs]
    payload = {"runs": rows, "dataset_summaries": summarise(rows)}

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "multi_dataset_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output / "multi_dataset_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["dataset_summaries"], indent=2))


if __name__ == "__main__":
    main()
