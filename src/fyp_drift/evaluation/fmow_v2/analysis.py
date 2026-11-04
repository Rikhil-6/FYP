"""Post-hoc measures from saved runs: weight-map stability across seeds, real-cloud scores, example maps.

Example::  python -m fyp_drift.evaluation.fmow_v2.analysis
"""

import glob
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_CACHE,
    DEFAULT_MANIFEST,
    load_rows,
    study_splits,
)
from scipy.stats import rankdata
from sklearn.metrics import f1_score


def main():
    R = Path(os.environ["FYP_RUNS"])
    OUT = R / "analysis"
    OUT.mkdir(exist_ok=True)
    rows, classes = load_rows(DEFAULT_MANIFEST)
    test = study_splits(rows)["future_test"]
    labels = np.array([r["class_id"] for r in test])
    res = {}
    rng = np.random.default_rng(0)
    sub = rng.choice(len(test), 1000, replace=False)
    stab = {}
    for f in sorted(glob.glob(str(R / "shift_test/*_s42_centred.npz"))):
        stem = Path(f).name.replace("_s42_centred.npz", "")
        ws = [
            np.load(str(R / f"shift_test/{stem}_s{s}_centred.npz"))["weights"][sub].astype(np.float32)
            for s in (42, 43, 44)
        ]
        rk = [np.apply_along_axis(rankdata, 1, w) for w in ws]
        corr = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            ra, rb = (rk[a] - rk[a].mean(1, keepdims=True), rk[b] - rk[b].mean(1, keepdims=True))
            corr.append(float(np.mean((ra * rb).sum(1) / np.sqrt((ra**2).sum(1) * (rb**2).sum(1)))))
        top = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            ta, tb = (np.argsort(-ws[a], 1)[:, :10], np.argsort(-ws[b], 1)[:, :10])
            top.append(float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(ta, tb, strict=True)])))
        stab[stem] = {"spearman": float(np.mean(corr)), "top10_overlap": float(np.mean(top))}
    res["stability"] = stab
    cloudy = np.array([float(r.get("cloud_cover", 0) or 0) >= 10 for r in test])
    res["real_cloud_n"] = int(cloudy.sum())

    def sub_f1(pred, mask):
        return float(
            f1_score(labels[mask], np.array(pred)[mask], labels=np.arange(classes), average="macro", zero_division=0)
        )

    rc = {}
    cands = {
        "frozen": [R / "main/arms/frozen_clean.json"],
        "lora_r8": sorted(R.glob("main/arms/clean_plain_l0.0_s*.json")),
        "lora_r2": sorted(R.glob("lora_rank_sweep/clean_plain_r2_s*.json")),
        "linear": sorted(R.glob("cloud_param/B_clean_attention_h0_s*.json")),
    }
    for arm in ("attention", "guided", "input", "gated", "prior", "gated_shuffled", "prior_shuffled"):
        cands[arm] = sorted(R.glob(f"saliency_attention/clean_{arm}_s[0-9]*.json"))
    for arm in ("soft_gated", "soft_prior", "soft_gated_shuffled"):
        cands[arm] = sorted(R.glob(f"saliency_soft/clean_{arm}_s[0-9]*.json"))
    for name, paths in cands.items():
        vals = []
        for p in paths:
            d = json.loads(Path(p).read_text())
            pred = d.get("predictions")
            if pred is None:
                continue
            vals.append((sub_f1(pred, cloudy), sub_f1(pred, ~cloudy)))
        if vals:
            rc[name] = {
                "seeds": len(vals),
                "real_cloudy_f1": float(np.mean([v[0] for v in vals])),
                "cloud_free_f1": float(np.mean([v[1] for v in vals])),
            }
    res["real_cloud"] = rc
    json.dump(res, open(OUT / "analysis.json", "w"), indent=1)
    sizes = json.loads(Path(DEFAULT_MANIFEST).with_name("image_sizes.json").read_text())

    def shifted_box(r):
        x0, y0, x1, y1 = r["crop"]
        side = x1 - x0
        cx, cy, bw, bh = (x0 + r["box_cx"] * side, y0 + r["box_cy"] * side, r["box_w"] * side, r["box_h"] * side)
        Wd, Hd = sizes[r["key"]]
        rng = random.Random(int(hashlib.sha256(r["key"].encode()).hexdigest()[:8], 16))
        bx, by = (min(0.5, bw / side / 2), min(0.5, bh / side / 2))
        best = (x0, y0)
        for _ in range(50):
            u, v = (rng.uniform(bx, 1 - bx), rng.uniform(by, 1 - by))
            if max(abs(u - 0.5), abs(v - 0.5)) < 0.15:
                continue
            nx0, ny0 = (min(max(cx - u * side, 0), Wd - side), min(max(cy - v * side, 0), Hd - side))
            if max(abs((cx - nx0) / side - 0.5), abs((cy - ny0) / side - 0.5)) >= 0.15:
                best = (nx0, ny0)
                break
        nx0, ny0 = best
        return [(cx - nx0) / side, (cy - ny0) / side, r["box_w"], r["box_h"]]

    cache = np.load(DEFAULT_CACHE, mmap_mode="r")
    scache = np.load("/mnt/sdd/FYP/cache/shift_test_v2/images_u8.npy", mmap_mode="r")
    W = {
        k: {
            v: np.load(str(R / f"shift_test/clean_{k}_s42_{v}.npz"))["weights"].astype(np.float32)
            for v in ("centred", "shifted")
        }
        for k in ("attention", "gated", "gated_shuffled")
    }
    P = {
        k: {v: np.load(str(R / f"shift_test/clean_{k}_s42_{v}.npz"))["predictions"] for v in ("centred", "shifted")}
        for k in ("gated",)
    }
    good = [
        i for i in range(len(test)) if P["gated"]["centred"][i] == labels[i] and P["gated"]["shifted"][i] == labels[i]
    ]
    pick = list(np.random.default_rng(3).choice(good, 6, replace=False))
    np.savez_compressed(
        OUT / "examples.npz",
        idx=np.array(pick),
        centred=np.stack([cache[test[i]["idx"]] for i in pick]),
        shifted=np.stack([scache[i] for i in pick]),
        **{f"{k}_{v}": W[k][v][pick] for k in W for v in ("centred", "shifted")},
        boxes=np.array([[test[i]["box_cx"], test[i]["box_cy"], test[i]["box_w"], test[i]["box_h"]] for i in pick]),
        classes=np.array([test[i]["category"] for i in pick]),
        sboxes=np.array([shifted_box(test[i]) for i in pick]),
    )
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
