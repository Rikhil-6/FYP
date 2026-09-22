"""Resumable CLEAR-100 download, extraction, and temporal manifest creation."""
import argparse
import csv
import json
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock, Timeout
from tqdm.auto import tqdm

SOURCE = "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/"
ARCHIVES = {"train": "clear100-train-image-only.zip", "test": "clear100-test.zip"}
ARCHIVE_ROOTS = {"train": "train_image_only", "test": "test"}


def split_for(source, bucket):
    if source == "train":
        if 0 <= bucket <= 6:
            return "train"
        return {7: "id_val", 8: "val", 9: "val"}.get(bucket, "unused")
    return {7: "id_test", 10: "test"}.get(bucket, "unused")


def download(root, source):
    archive = root / f"{source}.zip"
    if zipfile.is_zipfile(archive):
        print(f"Valid archive already present: {archive}", flush=True)
        return archive
    url = SOURCE + ARCHIVES[source]
    print(f"Downloading {url} -> {archive} (resumes partial file)", flush=True)
    subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
                    "--connect-timeout", "15", "--continue-at", "-", "--output", str(archive), url], check=True)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Downloaded archive is incomplete: {archive}")
    return archive


def extract(root, source, archive):
    marker = root / f"{source}.ready"
    archive_root = ARCHIVE_ROOTS[source]
    if marker.is_file() and (root / archive_root / "labeled_images").is_dir():
        print(f"{source} images already extracted; skipping.", flush=True)
        return
    with zipfile.ZipFile(archive) as handle:
        entries = handle.infolist()
        new_images = 0
        for item in tqdm(entries, desc=f"extract {source}", unit="file"):
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != archive_root:
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
    marker.write_text(f"extracted_utc={datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
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
                for path in class_dir.iterdir():
                    if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        candidates.append((path, source, bucket, class_dir.name))
    classes = sorted({item[3] for item in candidates})
    if len(classes) != 100:
        raise ValueError(f"Expected 100 CLEAR classes, found {len(classes)}")
    labels = {name: index for index, name in enumerate(classes)}
    counts = Counter()
    temp = root / "manifest.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "source", "bucket", "year", "class_name", "label", "split"))
        writer.writeheader()
        for path, source, bucket, class_name in candidates:
            split = split_for(source, bucket)
            counts[split] += 1
            writer.writerow({"path": path.relative_to(root).as_posix(), "source": source,
                             "bucket": bucket, "year": 2004 + bucket,
                             "class_name": class_name, "label": labels[class_name], "split": split})
    required = ("train", "id_val", "id_test", "val", "test")
    if any(counts[name] == 0 for name in required):
        raise ValueError(f"Required temporal split is empty: {dict(counts)}")
    temp.replace(root / "manifest.csv")
    (root / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    summary = {"dataset": "CLEAR-100", "source": SOURCE, "image_count": len(candidates),
               "class_count": len(classes), "counts": dict(counts),
               "time_mapping": "bucket 0=2004 through bucket 10=2014",
               "protocol": {"train": "train buckets 0-6", "id_val": "train bucket 7",
                            "id_test": "test bucket 7", "val": "train buckets 8-9",
                            "test": "test bucket 10"}}
    (root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    root = Path(args.data_root).resolve() / "clear100"
    root.mkdir(parents=True, exist_ok=True)
    print(f"CLEAR-100 destination: {root}", flush=True)
    try:
        with FileLock(str(root / ".ingest.lock"), timeout=0):
            if (root / "manifest.csv").is_file() and (root / "train.ready").is_file() and (root / "test.ready").is_file():
                print("CLEAR-100 is already ingested; skipping download and extraction.", flush=True)
                print((root / "dataset_summary.json").read_text(encoding="utf-8"), flush=True)
                return
            for source in ("train", "test"):
                archive = download(root, source)
                extract(root, source, archive)
            build_manifest(root)
    except Timeout as error:
        parser.exit(2, f"Another CLEAR-100 ingestion is already running: {error}\n")


if __name__ == "__main__":
    main()
