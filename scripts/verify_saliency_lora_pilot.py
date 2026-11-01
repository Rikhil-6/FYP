"""Verify paired IDs and report exact decision differences for LoRA arms."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(row["year"], row["scenario"], row["dataset_index"], row["true_class"])
            for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate evaluation ID in {path}")
    return keys, [row["predicted_class"] for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modes = ("none", "uniform", "shuffled", "saliency")
    loaded = {mode: read_rows(args.run_dir / f"{mode}-predictions.csv")
              for mode in modes}
    reference = loaded["none"][0]
    for mode in modes[1:]:
        if loaded[mode][0] != reference:
            raise ValueError(f"evaluation IDs differ for {mode}")
    digest = hashlib.sha256(json.dumps(reference, separators=(",", ":")).encode()).hexdigest()
    differences = {}
    for mode in modes[1:]:
        differences[mode] = {
            scenario: sum(
                a != b for key, a, b in zip(
                    reference, loaded["none"][1], loaded[mode][1]
                ) if key[1] == scenario
            ) for scenario in sorted({key[1] for key in reference})
        }
    result = {"rows": len(reference), "identical_ids": True,
              "id_sha256": digest, "prediction_differences_from_none": differences}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
