"""Grow the country-matched fMoW subset to 14k images per period, downloading images and metadata.

Example::  python -m fyp_drift.evaluation.fmow_v2.expand_subset
"""

import collections
import concurrent.futures
import csv
import json
import random
import time
import urllib.request
from pathlib import Path


def main():
    ROOT = Path("/mnt/sdd/FYP/data/fmow_v1.1")
    SUB = ROOT / "subset_7k_country_matched"
    OUT = ROOT / "subset_14k_country_matched"
    OUT.mkdir(exist_ok=True)
    URL = "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-rgb/"
    TARGET = 14000

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)

    def period(ts):
        y = int(ts[:4])
        return (
            "historical_2010_13"
            if 2010 <= y <= 2013
            else "adapt_2014_15" if 2014 <= y <= 2015 else "future_2016_17" if 2016 <= y <= 2017 else None
        )

    old = list(csv.DictReader(open(SUB / "manifest_downloaded.csv")))
    old_keys = {r["key"] for r in old}
    pool = {}
    for fn in ("metadata_sample_trainval_20k.jsonl", "metadata_train_remaining.jsonl", "metadata_sample_5000.jsonl"):
        for line in open(ROOT / fn):
            r = json.loads(line)
            p = period(r["timestamp"])
            if p and r["key"] not in pool:
                pool[r["key"]] = {**r, "period": p}
    P = ("historical_2010_13", "adapt_2014_15", "future_2016_17")
    avail = {p: collections.defaultdict(list) for p in P}
    for k, r in pool.items():
        if k not in old_keys:
            avail[r["period"]][r["country"]].append(k)
    have = {p: collections.Counter((r["country"] for r in old if r["period"] == p)) for p in P}
    countries = set.intersection(*[set(have[p]) | set(avail[p]) for p in P])
    cap = {c: min((have[p][c] + len(avail[p][c]) for p in P)) for c in countries}
    base = {c: max((have[p][c] for p in P)) for c in countries}
    lo, hi = (0.0, 1.0)
    for _ in range(60):
        f = (lo + hi) / 2
        q = {c: min(cap[c], max(base[c], int(f * cap[c]))) for c in countries}
        lo, hi = (f, hi) if sum(q.values()) < TARGET else (lo, f)
    q = {c: min(cap[c], max(base[c], int(hi * cap[c]))) for c in countries}
    log("quota total per period", sum(q.values()), "countries", len(q))
    rng = random.Random(0)
    new = []
    for p in P:
        for c, n in q.items():
            extra = n - have[p][c]
            if extra > 0:
                ks = sorted(avail[p][c])
                rng.shuffle(ks)
                new += [(k, p) for k in ks[:extra]]
    log("new images to fetch", len(new))

    def fetch(item):
        k, p = item
        r = pool[k]
        parts = k.split("/")
        dest = SUB / p / parts[1] / parts[2] / parts[3]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with urllib.request.urlopen(URL + k, timeout=60) as resp:
                data = resp.read()
            dest.write_bytes(data)
        with urllib.request.urlopen(URL + k.replace(".jpg", ".json"), timeout=60) as resp:
            meta = json.load(resp)
        W, H = (meta["img_width"], meta["img_height"])
        x, y, w, h = meta["bounding_boxes"][0]["box"]
        row = {
            "key": k,
            "split": r["split"],
            "category": r["category"],
            "timestamp": r["timestamp"],
            "country": r["country"],
            "utm": r["utm"],
            "period": p,
            "size_bytes": dest.stat().st_size,
            "head_error": "",
            "local_path": str(dest),
            "downloaded_bytes": dest.stat().st_size,
        }
        md = {
            "key": k,
            "cloud_cover": meta.get("cloud_cover"),
            "off_nadir_angle": meta.get("off_nadir_angle_dbl"),
            "sun_elevation": meta.get("sun_elevation_dbl"),
            "sun_azimuth": meta.get("sun_azimuth_dbl"),
            "target_azimuth": meta.get("target_azimuth_dbl"),
            "gsd": meta.get("gsd"),
            "box_cx": (x + w / 2) / W,
            "box_cy": (y + h / 2) / H,
            "box_w": w / W,
            "box_h": h / H,
        }
        return (row, md)

    rows, mds, fails, t0, nbytes = ([], [], [], time.time(), 0)
    with concurrent.futures.ThreadPoolExecutor(48) as ex:
        futs = {ex.submit(fetch, it): it for it in new}
        for i, fu in enumerate(concurrent.futures.as_completed(futs)):
            try:
                r, m = fu.result()
                rows.append(r)
                mds.append(m)
                nbytes += r["downloaded_bytes"]
            except Exception as e:
                fails.append((futs[fu][0], repr(e)))
            if (i + 1) % 1000 == 0:
                dt = time.time() - t0
                log(
                    f"{i + 1}/{len(new)} done, {nbytes / 1000000.0 / dt:.1f} MB/s, eta {(len(new) - i - 1) * dt / (i + 1) / 60:.1f} min, fails {len(fails)}"
                )
    old_md = list(csv.DictReader(open(SUB / "acquisition_metadata.csv")))
    fields = list(old[0].keys())
    mfields = list(old_md[0].keys())
    with open(OUT / "manifest_downloaded.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(old + rows)
    with open(OUT / "acquisition_metadata.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=mfields)
        w.writeheader()
        w.writerows(old_md + mds)
    with open(OUT / "failures.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["key", "error"])
        w.writerows(fails)
    cnt = collections.Counter((r["period"] for r in old + rows))
    log("DONE", dict(cnt), "fails", len(fails), f"{nbytes / 1000000000.0:.1f} GB in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
