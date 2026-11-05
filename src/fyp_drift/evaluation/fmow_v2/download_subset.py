"""Download the study subset (about 42k fMoW-RGB images) from the public bucket, starting from nothing.

The exact image list and each image's acquisition metadata are published in ``data_manifests/`` so no private files are
needed. Images are fetched over HTTPS (no AWS account) and checked against their published byte size.

Example::  python -m fyp_drift.evaluation.fmow_v2.download_subset [--root DIR] [--limit N] [--workers 32]

``--root`` defaults to ``$FYP_DATA`` (see scripts/fmow_v2/env.sh). The files written are
``<root>/subset_14k_country_matched/manifest_downloaded.csv`` and ``acquisition_metadata.csv`` (the two files
``FYP_MANIFEST`` and ``FYP_METADATA`` point to) and the images under ``<root>/subset_14k_country_matched/images/``.
"""

import argparse
import concurrent.futures
import csv
import gzip
import os
import time
import urllib.request
from pathlib import Path

URL = "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-rgb/"
FIELDS = ["key", "split", "category", "timestamp", "country", "utm", "period", "size_bytes", "head_error", "local_path", "downloaded_bytes"]


def read_gz(path):
    with gzip.open(path, "rt", newline="") as h:
        return list(csv.DictReader(h))


def fetch(row, dest, tries=3):
    """Download one image (skipped if already present with the right size); returns an error string or ''."""
    want = int(row["size_bytes"])
    if dest.exists() and dest.stat().st_size == want:
        return ""
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = ""
    for k in range(tries):
        try:
            with urllib.request.urlopen(URL + row["key"], timeout=60) as resp:
                data = resp.read()
            if len(data) != want:
                err = f"size {len(data)} != published {want}"
                continue
            dest.write_bytes(data)
            return ""
        except Exception as e:  # network errors are retried, then reported
            err = repr(e)
            time.sleep(1 + k)
    return err


def main():
    here = Path(__file__).resolve().parents[4] / "data_manifests"
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("FYP_DATA", "data/fmow_v1.1")))
    ap.add_argument("--manifests", type=Path, default=here)
    ap.add_argument("--limit", type=int, default=0, help="download only the first N images (for a quick test)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    out = a.root / "subset_14k_country_matched"
    rows = read_gz(a.manifests / "fmow_subset_manifest.csv.gz")
    meta = read_gz(a.manifests / "fmow_subset_acquisition_metadata.csv.gz")
    assert [r["key"] for r in rows] == [m["key"] for m in meta], "manifest and metadata lists differ"
    if a.limit:
        rows, meta = rows[: a.limit], meta[: a.limit]
    dests = [out / "images" / r["period"] / Path(r["key"]).parent.name / Path(r["key"]).name for r in rows]
    t0, bad = time.time(), []
    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        for i, err in enumerate(ex.map(fetch, rows, dests)):
            if err:
                bad.append((rows[i]["key"], err))
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(rows)} images, {len(bad)} failures, {time.time() - t0:.0f}s", flush=True)
    keep = [i for i, r in enumerate(rows) if r["key"] not in {b[0] for b in bad}]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest_downloaded.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS)
        w.writeheader()
        for i in keep:
            w.writerow({**rows[i], "head_error": "", "local_path": str(dests[i]), "downloaded_bytes": rows[i]["size_bytes"]})
    with open(out / "acquisition_metadata.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta[i] for i in keep)
    with open(out / "failures.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["key", "error"])
        w.writerows(bad)
    print(f"done: {len(keep)} images ok, {len(bad)} failed, {time.time() - t0:.0f}s; manifest in {out}")


if __name__ == "__main__":
    main()

