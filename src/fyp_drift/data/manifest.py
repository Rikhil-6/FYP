"""Schema checks for the CLEAR-100 temporal manifest."""

from collections import defaultdict
from pathlib import PurePosixPath


NUM_CLASSES = 100
EXPECTED_BUCKETS = set(range(11))
ARCHIVE_ROOTS = {"train": "train_image_only", "test": "test"}


def expected_split(source, bucket):
    if source == "train":
        if 0 <= bucket <= 6:
            return "train"
        return {7: "id_val", 8: "val", 9: "val"}.get(bucket, "unused")
    if source == "test":
        return {7: "id_test", 10: "test"}.get(bucket, "unused")
    raise ValueError(f"Unknown CLEAR-100 source: {source}")


def validate_manifest_rows(rows):
    """Reject temporal leakage, incomplete buckets, and inconsistent metadata."""
    required = {"path", "source", "bucket", "year", "class_name", "label", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CLEAR-100 manifest is empty or missing required columns")

    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("CLEAR-100 manifest contains duplicate image paths")
    image_ids = [PurePosixPath(path).stem for path in paths]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(
            "CLEAR-100 manifest reuses an image identifier across temporal rows"
        )

    class_to_label = {}
    labels_by_bucket = defaultdict(set)
    buckets_by_source = defaultdict(set)
    for row in rows:
        source = row["source"]
        bucket = int(row["bucket"])
        year = int(row["year"])
        label = int(row["label"])
        if source not in ARCHIVE_ROOTS:
            raise ValueError(f"Unknown CLEAR-100 source: {source}")
        if bucket not in EXPECTED_BUCKETS:
            raise ValueError(f"CLEAR-100 bucket outside 0-10: {bucket}")
        if year != 2004 + bucket:
            raise ValueError(
                f"CLEAR-100 year/bucket mismatch for {row['path']}: {year}, {bucket}"
            )
        if not 0 <= label < NUM_CLASSES:
            raise ValueError(f"CLEAR-100 label outside 0-{NUM_CLASSES - 1}: {label}")
        if row["split"] != expected_split(source, bucket):
            raise ValueError(
                f"CLEAR-100 split mismatch for {row['path']}: expected "
                f"{expected_split(source, bucket)}, found {row['split']}"
            )

        parts = PurePosixPath(row["path"]).parts
        expected_prefix = (
            ARCHIVE_ROOTS[source],
            "labeled_images",
            str(bucket),
            row["class_name"],
        )
        if len(parts) != 5 or parts[:4] != expected_prefix:
            raise ValueError(
                f"CLEAR-100 path disagrees with its metadata: {row['path']}"
            )

        previous = class_to_label.setdefault(row["class_name"], label)
        if previous != label:
            raise ValueError(f"Class {row['class_name']} maps to multiple labels")
        buckets_by_source[source].add(bucket)
        labels_by_bucket[(source, bucket)].add(label)

    if len(class_to_label) != NUM_CLASSES or set(class_to_label.values()) != set(
        range(NUM_CLASSES)
    ):
        raise ValueError("CLEAR-100 manifest must map 100 classes one-to-one to labels")
    for source in ARCHIVE_ROOTS:
        if buckets_by_source[source] != EXPECTED_BUCKETS:
            missing = sorted(EXPECTED_BUCKETS - buckets_by_source[source])
            extra = sorted(buckets_by_source[source] - EXPECTED_BUCKETS)
            raise ValueError(
                f"CLEAR-100 {source} buckets are incomplete; missing={missing}, extra={extra}"
            )
        for bucket in EXPECTED_BUCKETS:
            if labels_by_bucket[(source, bucket)] != set(range(NUM_CLASSES)):
                missing = sorted(
                    set(range(NUM_CLASSES)) - labels_by_bucket[(source, bucket)]
                )
                raise ValueError(
                    f"CLEAR-100 {source} bucket {bucket} is missing classes: {missing}"
                )
