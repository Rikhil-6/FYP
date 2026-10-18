"""Audit the installed CLEAR-100 manifest before an experiment starts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from fyp_drift.data.clear100 import NUM_CLASSES, validate_manifest_rows
from fyp_drift.data.ingestion import ARCHIVE_SHA256, archive_sha256


def audit_manifest(data_root, check_files=False, check_archives=False):
    root = Path(data_root).resolve() / "clear100"
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"CLEAR-100 manifest missing: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_manifest_rows(rows)

    split_counts = Counter(row["split"] for row in rows)
    bucket_counts = Counter(
        (row["source"], int(row["bucket"]), int(row["year"])) for row in rows
    )
    classes_by_bucket = defaultdict(set)
    for row in rows:
        classes_by_bucket[(row["source"], int(row["bucket"]))].add(
            int(row["label"])
        )

    missing = []
    if check_files:
        missing = [row["path"] for row in rows if not (root / row["path"]).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest images are missing; first path: {missing[0]}"
            )

    archive_checks = {}
    if check_archives:
        for source, expected in ARCHIVE_SHA256.items():
            path = root / f"{source}.zip"
            if not path.is_file():
                raise FileNotFoundError(f"CLEAR-100 archive missing: {path}")
            actual = archive_sha256(path)
            archive_checks[source] = {
                "path": str(path),
                "sha256": actual,
                "matches_expected": actual == expected,
            }
            if actual != expected:
                raise ValueError(f"CLEAR-100 {source} archive checksum mismatch")

    return {
        "dataset": "CLEAR-100",
        "manifest": str(manifest),
        "rows": len(rows),
        "classes": NUM_CLASSES,
        "unique_paths": len({row["path"] for row in rows}),
        "split_counts": dict(sorted(split_counts.items())),
        "buckets": [
            {
                "source": source,
                "bucket": bucket,
                "year": year,
                "images": count,
                "represented_classes": len(classes_by_bucket[(source, bucket)]),
            }
            for (source, bucket, year), count in sorted(bucket_counts.items())
        ],
        "files_checked": bool(check_files),
        "missing_files": len(missing),
        "archive_checks": archive_checks,
        "checks": {
            "unique_paths": True,
            "year_equals_2004_plus_bucket": True,
            "source_bucket_split_mapping": True,
            "one_to_one_class_labels": True,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate CLEAR-100 paths, labels, years, buckets, and splits."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--check-archives", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_manifest(
        args.data_root,
        check_files=args.check_files,
        check_archives=args.check_archives,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
