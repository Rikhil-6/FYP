"""Evaluate saved LoRA pilot arms on identical full CLEAR year/scenario rows."""

import argparse
import csv
import json
from pathlib import Path

import torch

from fyp_drift.data.clear100 import open_dataset, year_loader
from fyp_drift.data.corruptions import drift_transform
from fyp_drift.training.saliency_lora_pilot import build_model, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--scenarios", nargs="+", default=(
        "clean", "viewpoint_shift", "occlusion", "realistic_mix"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    config = json.loads((args.run_dir / "config.json").read_text())
    seed = int(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = open_dataset(config["data_root"])
    args.output.mkdir(parents=True)
    report = {"source_run": str(args.run_dir), "year": args.year,
              "scenarios": list(args.scenarios), "arms": {}}
    for mode in config["modes"]:
        model = build_model(Path(config["base_checkpoint"]), device, config["rank"])
        model.load_state_dict(torch.load(args.run_dir / f"{mode}.pt",
                                         map_location=device, weights_only=True))
        scores = {}
        with (args.output / f"{mode}-predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("year", "scenario", "dataset_index", "true_class", "predicted_class"))
            for scenario in args.scenarios:
                loader, _ = year_loader(
                    dataset, "test_source", (args.year,), args.batch_size,
                    seed=seed,
                    image_transform=drift_transform(scenario, max(1, args.year - 2011), seed),
                )
                metrics, rows = evaluate(model, loader, device)
                scores[scenario] = {"count": metrics["count"],
                                    "accuracy": metrics["accuracy"],
                                    "macro_f1": metrics["macro_f1"]}
                writer.writerows((args.year, scenario, *row) for row in rows)
                print(json.dumps({"mode": mode, "scenario": scenario,
                                  "scores": scores[scenario]}), flush=True)
        report["arms"][mode] = scores
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
