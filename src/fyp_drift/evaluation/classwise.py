"""Export readable class-level comparisons from router decision logs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from fyp_drift.evaluation.metrics import classwise_model_comparison


def _read_decisions(path):
    integer_fields = {
        "severity",
        "dataset_index",
        "year",
        "true_class",
        "old_class",
        "new_class",
        "routed_class",
    }
    boolean_fields = {"needs_update", "routed_to_new"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in integer_fields & row.keys():
            row[field] = int(row[field])
        for field in boolean_fields & row.keys():
            row[field] = row[field].lower() == "true"
    return rows


def _read_class_names(path):
    if path is None:
        return None
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Class names must be a JSON list of strings")
    return names


def build_report(decisions, class_names=None, min_support=20):
    groups = defaultdict(list)
    for row in decisions:
        scenario = row.get("scenario", "clean")
        groups[(scenario, str(row["year"]))].append(row)
        groups[(scenario, "all")].append(row)

    report = {
        "metric_definition": "One-versus-rest F1 and IoU for each true class",
        "stratification": (
            "Classes with sufficient support are ranked by historical F1 and divided "
            "into four equally sized groups from weakest to strongest"
        ),
        "groups": {},
    }
    for (scenario, year), rows in sorted(groups.items()):
        comparison = classwise_model_comparison(rows, min_support=min_support)
        if class_names:
            for class_row in comparison["classes"]:
                class_row["class_name"] = class_names[class_row["class_id"]]
        report["groups"][f"{scenario}:{year}"] = {
            "scenario": scenario,
            "year": year,
            "sample_count": len(rows),
            **comparison,
        }
    return report


def write_tables(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        **report,
        "groups": {
            name: {key: value for key, value in group.items() if key != "classes"}
            for name, group in report["groups"].items()
        },
    }
    (output_dir / "classwise_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    class_fields = [
        "scenario",
        "year",
        "class_id",
        "class_name",
        "support",
        "sufficient_support",
        "old_f1",
        "new_f1",
        "routed_f1",
        "new_f1_delta_pp",
        "routed_f1_delta_pp",
        "old_iou",
        "new_iou",
        "routed_iou",
        "new_iou_delta_pp",
        "routed_iou_delta_pp",
        "route_rate",
        "helpful_routes",
        "harmful_routes",
        "beneficial_available",
        "beneficial_missed",
        "harmful_available",
        "harmful_avoided",
    ]
    with (output_dir / "classwise_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=class_fields)
        writer.writeheader()
        for group in report["groups"].values():
            for row in group["classes"]:
                routing = row.get("routing", {})
                writer.writerow(
                    {
                        "scenario": group["scenario"],
                        "year": group["year"],
                        "class_id": row["class_id"],
                        "class_name": row.get("class_name", ""),
                        "support": row["support"],
                        "sufficient_support": row["sufficient_support"],
                        "old_f1": row["models"]["old"]["f1"],
                        "new_f1": row["models"]["new"]["f1"],
                        "routed_f1": row["models"]["routed"]["f1"],
                        "new_f1_delta_pp": row["f1_delta_pp"]["new"],
                        "routed_f1_delta_pp": row["f1_delta_pp"]["routed"],
                        "old_iou": row["models"]["old"]["iou"],
                        "new_iou": row["models"]["new"]["iou"],
                        "routed_iou": row["models"]["routed"]["iou"],
                        "new_iou_delta_pp": row["iou_delta_pp"]["new"],
                        "routed_iou_delta_pp": row["iou_delta_pp"]["routed"],
                        "route_rate": routing.get("route_rate", ""),
                        "helpful_routes": routing.get("helpful", ""),
                        "harmful_routes": routing.get("harmful", ""),
                        "beneficial_available": routing.get("beneficial_available", ""),
                        "beneficial_missed": routing.get("beneficial_missed", ""),
                        "harmful_available": routing.get("harmful_available", ""),
                        "harmful_avoided": routing.get("harmful_avoided", ""),
                    }
                )

    strata_fields = [
        "scenario",
        "year",
        "stratum",
        "description",
        "class_count",
        "historical_f1_min",
        "historical_f1_max",
        "old_mean_f1",
        "new_mean_f1",
        "routed_mean_f1",
        "new_mean_f1_delta_pp",
        "routed_mean_f1_delta_pp",
    ]
    with (output_dir / "classwise_strata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=strata_fields)
        writer.writeheader()
        for group in report["groups"].values():
            for row in group["historical_f1_rank_strata"]:
                writer.writerow(
                    {
                        "scenario": group["scenario"],
                        "year": group["year"],
                        "stratum": row["stratum"],
                        "description": row["description"],
                        "class_count": row["class_count"],
                        "historical_f1_min": row["historical_f1_range"][0],
                        "historical_f1_max": row["historical_f1_range"][1],
                        "old_mean_f1": row["mean_f1"]["old"],
                        "new_mean_f1": row["mean_f1"]["new"],
                        "routed_mean_f1": row["mean_f1"]["routed"],
                        "new_mean_f1_delta_pp": row["mean_f1_delta_pp"]["new"],
                        "routed_mean_f1_delta_pp": row["mean_f1_delta_pp"]["routed"],
                    }
                )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-names", type=Path)
    parser.add_argument("--minimum-support", type=int, default=20)
    args = parser.parse_args(argv)

    decisions = _read_decisions(args.decisions)
    report = build_report(
        decisions,
        class_names=_read_class_names(args.class_names),
        min_support=args.minimum_support,
    )
    write_tables(report, args.output_dir)
    print(args.output_dir / "classwise_summary.json")


if __name__ == "__main__":
    main()
