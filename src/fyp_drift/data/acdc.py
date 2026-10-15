"""Build a manifest for the paired ACDC adverse-driving dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CONDITIONS = ("fog", "night", "rain", "snow")
SPLITS = ("train", "val", "test")


def paired_paths(image_path: Path, root: Path, condition: str, split: str):
    recording = image_path.parent.name
    reference_name = image_path.name.replace("_rgb_anon.png", "_rgb_ref_anon.png")
    label_name = image_path.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
    invalid_name = image_path.name.replace("_rgb_anon.png", "_gt_invIds.png")
    return {
        "reference": root
        / "rgb_anon"
        / condition
        / f"{split}_ref"
        / recording
        / reference_name,
        "label": root / "gt" / condition / split / recording / label_name,
        "invalid_mask": root / "gt" / condition / split / recording / invalid_name,
    }


def scan_acdc(root: Path, splits=("train", "val"), require_labels=True):
    """Return paired ACDC rows and fail on incomplete normal/adverse pairs."""
    root = Path(root)
    if not (root / "rgb_anon").is_dir():
        raise FileNotFoundError(
            f"ACDC images were not found beneath {root}. Download and extract "
            "rgb_anon_trainvaltest.zip from the registered ACDC website first."
        )

    rows = []
    missing = []
    for condition in CONDITIONS:
        for split in splits:
            if split not in SPLITS:
                raise ValueError(f"Unknown ACDC split: {split}")
            image_root = root / "rgb_anon" / condition / split
            if not image_root.is_dir():
                missing.append(str(image_root))
                continue
            for image_path in sorted(image_root.glob("*/*_rgb_anon.png")):
                related = paired_paths(image_path, root, condition, split)
                required = [related["reference"]]
                if require_labels and split != "test":
                    required.extend((related["label"], related["invalid_mask"]))
                absent = [str(path) for path in required if not path.is_file()]
                if absent:
                    missing.extend(absent)
                    continue
                rows.append(
                    {
                        "pair_id": f"{condition}/{split}/{image_path.parent.name}/{image_path.stem}",
                        "condition": condition,
                        "split": split,
                        "recording": image_path.parent.name,
                        "adverse_image": str(image_path.resolve()),
                        "normal_image": str(related["reference"].resolve()),
                        "label": str(related["label"].resolve())
                        if related["label"].is_file()
                        else "",
                        "invalid_mask": str(related["invalid_mask"].resolve())
                        if related["invalid_mask"].is_file()
                        else "",
                    }
                )

    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"ACDC is incomplete; {len(missing)} required paths are missing. "
            f"First entries:\n{preview}"
        )
    if not rows:
        raise ValueError(f"No ACDC image pairs were found beneath {root}")
    return rows


def write_manifest(rows, output: Path):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate ACDC normal/adverse pairs and write a manifest."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ACDC"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=("train", "val")
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Permit image-only test data without ground-truth files.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or args.data_root / "paired_manifest.csv"
    rows = scan_acdc(
        args.data_root,
        splits=args.splits,
        require_labels=not args.allow_missing_labels,
    )
    write_manifest(rows, output)
    counts = Counter((row["split"], row["condition"]) for row in rows)
    summary = {
        "dataset": "ACDC adverse conditions driving dataset",
        "root": str(args.data_root.resolve()),
        "manifest": str(output.resolve()),
        "pairs": len(rows),
        "counts": {
            f"{split}:{condition}": count
            for (split, condition), count in sorted(counts.items())
        },
        "paired_fields": [
            "adverse_image",
            "normal_image",
            "label",
            "invalid_mask",
        ],
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
