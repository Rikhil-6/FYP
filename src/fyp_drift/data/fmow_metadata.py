"""Per-image fMoW acquisition metadata: real cloud cover, viewing and sun geometry, resolution, object box.

The fMoW-RGB release stores one JSON beside every image. This module fetches those JSONs
for a downloaded manifest and writes one CSV row per image, so real acquisition nuisances
can be used as evaluation strata instead of synthetic edits.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import urllib.request
from pathlib import Path

FMOW_RGB = "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-rgb/"
FIELDS = ("key", "cloud_cover", "off_nadir_angle", "sun_elevation", "sun_azimuth", "target_azimuth", "gsd",
          "box_cx", "box_cy", "box_w", "box_h")


def acquisition_row(key):
    """One metadata row for a manifest key; box values are fractions of image width/height."""
    with urllib.request.urlopen(FMOW_RGB + key.replace(".jpg", ".json"), timeout=30) as response:
        meta = json.load(response)
    width, height = meta["img_width"], meta["img_height"]
    x, y, w, h = meta["bounding_boxes"][0]["box"]
    return {"key": key, "cloud_cover": meta.get("cloud_cover"), "off_nadir_angle": meta.get("off_nadir_angle_dbl"),
            "sun_elevation": meta.get("sun_elevation_dbl"), "sun_azimuth": meta.get("sun_azimuth_dbl"),
            "target_azimuth": meta.get("target_azimuth_dbl"), "gsd": meta.get("gsd"),
            "box_cx": (x + w / 2) / width, "box_cy": (y + h / 2) / height, "box_w": w / width, "box_h": h / height}


def write_metadata(manifest, output, threads=32):
    with open(manifest, newline="", encoding="utf-8") as handle:
        keys = [row["key"] for row in csv.DictReader(handle) if row.get("local_path")]
    rows, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        futures = {pool.submit(acquisition_row, key): key for key in keys}
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:  # one retry path: record and continue; missing rows are reported
                failures.append((futures[future], repr(exc)))
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["key"]))
    return len(rows), failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/mnt/sdd/FYP/data/fmow_v1.1/subset_7k_country_matched/manifest_downloaded.csv"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/sdd/FYP/data/fmow_v1.1/subset_7k_country_matched/acquisition_metadata.csv"))
    args = parser.parse_args(argv)
    written, failures = write_metadata(args.manifest, args.output)
    print(f"wrote {written} rows to {args.output}; {len(failures)} failures", flush=True)
    for key, error in failures[:10]:
        print("failed", key, error)


if __name__ == "__main__":
    main()
