"""Resumable CLEAR-100 download, extraction, and temporal manifest creation."""

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock, Timeout
from tqdm.auto import tqdm

from fyp_drift.data.manifest import (
    ARCHIVE_ROOTS,
    NUM_CLASSES,
    expected_split,
    validate_manifest_rows,
)

SOURCE = "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/"
ARCHIVES = {"train": "clear100-train-image-only.zip", "test": "clear100-test.zip"}
ARCHIVE_SHA256 = {
    "train": "0376b952674e6ef55c3923ee4ce61e5b299fea4e29bbc4780530636e8988fd72",
    "test": "c939753be4e62dc7732347e5e636ea599022c82f45443ea9e7166167e467abd0",
}


def split_for(source, bucket):
    return expected_split(source, bucket)


def archive_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path, source):
    if not zipfile.is_zipfile(path):
        return False
    actual = archive_sha256(path)
    if actual != ARCHIVE_SHA256[source]:
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {ARCHIVE_SHA256[source]}, "
            f"found {actual}. Remove the archive and run ingestion again."
        )
    return True


def download(root, source):
    archive = root / f"{source}.zip"
    if verify_archive(archive, source):
        print(f"Valid archive already present: {archive}", flush=True)
        return archive
    url = SOURCE + ARCHIVES[source]
    print(f"Downloading {url} -> {archive} (resumes partial file)", flush=True)
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "15",
            "--continue-at",
            "-",
            "--output",
            str(archive),
            url,
        ],
        check=True,
    )
    if not verify_archive(archive, source):
        raise RuntimeError(f"Downloaded archive is incomplete: {archive}")
    return archive


def extract(root, source, archive, force=False):
    marker = root / f"{source}.ready"
    archive_root = ARCHIVE_ROOTS[source]
    if (
        not force
        and marker.is_file()
        and (root / archive_root / "labeled_images").is_dir()
    ):
        print(f"{source} images already extracted; skipping.", flush=True)
        return
    with zipfile.ZipFile(archive) as handle:
        entries = handle.infolist()
        new_images = 0
        for item in tqdm(entries, desc=f"extract {source}", unit="file"):
            path = Path(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != archive_root
            ):
                raise ValueError(f"Unexpected archive path: {item.filename}")
            destination = root / path
            if item.is_dir():
                continue
            if destination.is_file() and destination.stat().st_size == item.file_size:
                continue
            handle.extract(item, root)
            if destination.suffix.lower() in (".jpg", ".jpeg", ".png"):
                new_images += 1
    if not (root / archive_root / "labeled_images").is_dir():
        raise RuntimeError(f"{source} archive contains no labeled_images folder")
    marker.write_text(
        f"extracted_utc={datetime.now(UTC).isoformat()}\n", encoding="utf-8"
    )
    print(f"{source} extraction ready; {new_images} new images", flush=True)


def build_manifest(root):
    candidates = []
    for source in ("train", "test"):
        image_root = root / ARCHIVE_ROOTS[source] / "labeled_images"
        for bucket_dir in sorted(image_root.iterdir()):
            if not bucket_dir.is_dir() or not bucket_dir.name.isdigit():
                continue
            bucket = int(bucket_dir.name)
            for class_dir in sorted(bucket_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                for path in sorted(class_dir.iterdir()):
                    if path.is_file() and path.suffix.lower() in (
                        ".jpg",
                        ".jpeg",
                        ".png",
                    ):
                        candidates.append((path, source, bucket, class_dir.name))
    candidates.sort(key=lambda item: item[0].relative_to(root).as_posix())
    classes = sorted({item[3] for item in candidates})
    if len(classes) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} CLEAR classes, found {len(classes)}")
    labels = {name: index for index, name in enumerate(classes)}
    counts = Counter()
    rows = []
    for path, source, bucket, class_name in candidates:
        split = split_for(source, bucket)
        counts[split] += 1
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "source": source,
                "bucket": str(bucket),
                "year": str(2004 + bucket),
                "class_name": class_name,
                "label": str(labels[class_name]),
                "split": split,
            }
        )
    validate_manifest_rows(rows)

    temp = root / "manifest.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path",
                "source",
                "bucket",
                "year",
                "class_name",
                "label",
                "split",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    required = ("train", "id_val", "id_test", "val", "test")
    if any(counts[name] == 0 for name in required):
        raise ValueError(f"Required temporal split is empty: {dict(counts)}")
    temp.replace(root / "manifest.csv")
    (root / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    summary = {
        "dataset": "CLEAR-100",
        "source": SOURCE,
        "image_count": len(candidates),
        "class_count": len(classes),
        "counts": dict(counts),
        "time_mapping": "bucket 0=2004 through bucket 10=2014",
        "protocol": {
            "train": "train buckets 0-6",
            "id_val": "train bucket 7",
            "id_test": "test bucket 7",
            "val": "train buckets 8-9",
            "test": "test bucket 10",
        },
    }
    (root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def archive_image_inventory(archive):
    """Return archive image paths and byte sizes keyed by relative path."""
    with zipfile.ZipFile(archive) as handle:
        return {
            item.filename: item.file_size
            for item in handle.infolist()
            if not item.is_dir()
            and Path(item.filename).suffix.lower() in (".jpg", ".jpeg", ".png")
            and "labeled_images" in Path(item.filename).parts
        }


def validate_installed_dataset(root):
    """Verify the manifest against both archives and the extracted files."""
    manifest = root / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_manifest_rows(rows)

    manifest_paths = {row["path"] for row in rows}
    inventory = {}
    for source in ("train", "test"):
        inventory.update(archive_image_inventory(root / f"{source}.zip"))
    archive_paths = set(inventory)
    if manifest_paths != archive_paths:
        missing = sorted(archive_paths - manifest_paths)
        extra = sorted(manifest_paths - archive_paths)
        raise ValueError(
            "CLEAR-100 manifest differs from the verified archives; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    invalid_files = []
    for relative_path, expected_size in inventory.items():
        path = root / relative_path
        if not path.is_file() or path.stat().st_size != expected_size:
            invalid_files.append(relative_path)
            if len(invalid_files) == 3:
                break
    if invalid_files:
        raise ValueError(
            "CLEAR-100 extracted files are missing or have the wrong size: "
            f"{invalid_files}"
        )
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    root = Path(args.data_root).resolve() / "clear100"
    root.mkdir(parents=True, exist_ok=True)
    print(f"CLEAR-100 destination: {root}", flush=True)
    try:
        with FileLock(str(root / ".ingest.lock"), timeout=0):
            if (
                (root / "manifest.csv").is_file()
                and (root / "train.ready").is_file()
                and (root / "test.ready").is_file()
            ):
                for source in ("train", "test"):
                    if not verify_archive(root / f"{source}.zip", source):
                        raise RuntimeError(
                            f"CLEAR-100 {source} archive is missing or incomplete"
                        )
                try:
                    rows = validate_installed_dataset(root)
                except (OSError, ValueError) as error:
                    print(f"Installed data need repair: {error}", flush=True)
                    for source in ("train", "test"):
                        extract(root, source, root / f"{source}.zip", force=True)
                    build_manifest(root)
                    rows = validate_installed_dataset(root)
                print(
                    f"CLEAR-100 is already valid ({rows} images); skipping download.",
                    flush=True,
                )
                if (root / "dataset_summary.json").is_file():
                    print(
                        (root / "dataset_summary.json").read_text(encoding="utf-8"),
                        flush=True,
                    )
                return
            for source in ("train", "test"):
                archive = download(root, source)
                extract(root, source, archive)
            build_manifest(root)
            validate_installed_dataset(root)
    except Timeout as error:
        parser.exit(2, f"Another CLEAR-100 ingestion is already running: {error}\n")


if __name__ == "__main__":
    main()
