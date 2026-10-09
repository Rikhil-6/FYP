"""Create a compact, reviewable comparison from completed router runs."""

import argparse
import json
from pathlib import Path


def read_events(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if "epoch" in event and "adapt_loss" in event:
                rows.append(event)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=RUN_DIR; repeat for every run",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"metric": "macro_f1", "runs": {}}
    controls = (
        "epochs",
        "batch_size",
        "seed",
        "adapt_year",
        "calibration_year",
        "test_years",
        "long_horizon",
    )
    for specification in args.run:
        name, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        config = json.loads((path / "config.json").read_text())
        drift = json.loads((path / "metrics" / "future_drift.json").read_text())
        policy = json.loads((path / "metrics" / "router_policy.json").read_text())
        scenarios = {}
        for scenario, years in drift["scenarios"].items():
            scenarios[scenario] = {}
            for year, payload in years.items():
                result = payload["results"]["all"]
                scenarios[scenario][year] = {
                    key: result[key]["macro_f1"] for key in ("old", "new", "routed")
                } | {
                    key: result[key]
                    for key in ("helpful_routes", "harmful_routes", "routed_to_new")
                }
        report["runs"][name] = {
            "path": str(path),
            "config": {
                key: config.get(key)
                for key in controls + ("adaptation_profile", "disable_saliency_loss")
            },
            "epochs": read_events(path / "logs" / "events.jsonl"),
            "policy": policy,
            "scenarios": scenarios,
        }
    report["comparability"] = {
        key: {
            "matched": len(
                {
                    json.dumps(run["config"].get(key), sort_keys=True)
                    for run in report["runs"].values()
                }
            )
            == 1,
            "values": {
                name: run["config"].get(key) for name, run in report["runs"].items()
            },
        }
        for key in controls
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
