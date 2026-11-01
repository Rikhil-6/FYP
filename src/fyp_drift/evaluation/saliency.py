"""Summarize whether transparent saliency features correspond to useful routes."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FEATURES = (
    "saliency_js",
    "confidence_gain",
    "entropy_reduction",
    "evidence_gain",
    "route_probability",
)


def outcome(row):
    old_ok = int(row["old_class"]) == int(row["true_class"])
    new_ok = int(row["new_class"]) == int(row["true_class"])
    routed = row["routed_to_new"].lower() == "true"
    if routed and not old_ok and new_ok:
        return "helpful"
    if routed and old_ok and not new_ok:
        return "harmful"
    if routed:
        return "neutral_routed"
    return "kept_old"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.run_dir / "logs" / "drift_routing_decisions.csv"
    if not source.is_file():
        source = args.run_dir / "logs" / "routing_decisions.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        row["outcome"] = outcome(row)
        grouped[
            (row.get("scenario", "clean"), int(row["year"]), row["outcome"])
        ].append(row)
    summary = []
    for (scenario, year, result), values in sorted(grouped.items()):
        entry = {
            "scenario": scenario,
            "year": year,
            "outcome": result,
            "count": len(values),
        }
        for feature in FEATURES:
            numbers = np.asarray([float(row[feature]) for row in values])
            entry[f"mean_{feature}"] = round(float(numbers.mean()), 6)
            entry[f"median_{feature}"] = round(float(np.median(numbers)), 6)
        summary.append(entry)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "saliency_route_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output / "saliency_route_analysis.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    combined_2014 = [
        row
        for row in rows
        if row.get("scenario") == "combined" and int(row["year"]) == 2014
    ]
    labels = ("helpful", "harmful", "neutral_routed", "kept_old")
    headline = {
        "scenario": "combined",
        "year": 2014,
        "rows": len(combined_2014),
        "outcomes": {},
    }
    for label in labels:
        selected = [row for row in combined_2014 if outcome(row) == label]
        headline["outcomes"][label] = {
            "count": len(selected),
            **(
                {
                    f"mean_{feature}": round(
                        float(np.mean([float(row[feature]) for row in selected])), 6
                    )
                    for feature in FEATURES
                }
                if selected
                else {}
            ),
        }
    helpful = headline["outcomes"]["helpful"]["count"]
    harmful = headline["outcomes"]["harmful"]["count"]
    headline["helpful_to_harmful_ratio"] = (
        round(helpful / harmful, 4) if harmful else None
    )
    (args.output / "saliency_route_headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, feature, title in zip(
        axes,
        ("saliency_js", "route_probability"),
        ("Saliency divergence", "Logistic routing score"),
    ):
        values = [
            [float(row[feature]) for row in combined_2014 if outcome(row) == label]
            for label in labels
        ]
        axis.boxplot(
            values,
            tick_labels=[label.replace("_", "\n") for label in labels],
            showfliers=False,
        )
        axis.set_title(f"2014 combined drift: {title}")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output / "saliency_route_outcomes.png", dpi=180)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "groups": len(summary),
                "output": str(args.output),
                "combined_2014": headline,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
