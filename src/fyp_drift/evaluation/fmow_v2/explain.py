"""Exact explanations: per-cell maps of the old and new heads over the test set, then 20x20 patch maps for the examples.

Example::  python -m fyp_drift.evaluation.fmow_v2.explain            # both steps; or: explain maps / explain patches
"""

import argparse
import json
import os
import types
from pathlib import Path

import numpy as np
import torch

from fyp_drift.evaluation import saliency_attention_study as sas
from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_CACHE,
    DEFAULT_MANIFEST,
    load_rows,
    study_splits,
)


def maps():
    R = Path(os.environ["FYP_RUNS"])
    T = Path(os.environ["FYP_TOKENS"])
    OUT = R / "analysis"
    rows, classes = load_rows(DEFAULT_MANIFEST)
    sp = study_splits(rows)
    test = sp["future_test"]
    n = len(test)
    labels = np.array([r["class_id"] for r in test])
    names = {r["class_id"]: r["category"] for r in rows}
    boxes = np.array([[r["box_cx"], r["box_cy"], r["box_w"], r["box_h"]] for r in test])
    hs = torch.load(R / "main/head.pt", weights_only=True)
    W = hs["weight"].float().cuda()
    b = hs["bias"].float().cuda()

    def inbox(G):
        c = (np.arange(G) + 0.5) / G
        X, Y = np.meshgrid(c, c)
        return np.stack(
            [
                ((np.abs(X - cx) <= w / 2) & (np.abs(Y - cy) <= h / 2)).ravel()
                for cx, cy, w, h in boxes
            ]
        )

    def metrics(ev, mask):
        """ev: (n, N) signed contributions. Share of positive evidence in box, lift over box area, share of cells holding half of it."""
        pos = np.clip(ev, 0, None)
        tot = pos.sum(1) + 1e-09
        inside = (pos * mask).sum(1) / tot
        area = mask.mean(1).clip(0.001)
        srt = -np.sort(-pos, 1)
        cum = srt.cumsum(1) / tot[:, None]
        half = (cum < 0.5).sum(1) + 1
        return (inside, inside / area, half / ev.shape[1])

    res, keep = ({}, {})
    for G, path in (
        (20, Path(os.environ.get("FYP_ROOT", "/mnt/sdd/FYP")) / "cache/adapter_tokens_v2/clean_future_test.npy"),
        (10, T / "clean_future_test.npy"),
    ):
        tok = np.load(path, mmap_mode="r")
        EV, EVc, PR = ([], [], [])
        with torch.no_grad():
            for s in range(0, n, 128):
                h = torch.from_numpy(
                    np.asarray(tok[s : s + 128], dtype=np.float32)
                ).cuda()
                e = h @ W.T
                L = e.mean(1) + b
                p = L.argmax(1)
                r2 = L.topk(2, 1).indices
                ru = torch.where(r2[:, 0] == p, r2[:, 1], r2[:, 0])
                ar = torch.arange(len(p), device=h.device)
                EV.append((e[ar, :, p] - e.mean(2)).cpu().numpy() / e.shape[1])
                EVc.append((e[ar, :, p] - e[ar, :, ru]).cpu().numpy() / e.shape[1])
                PR.append(p.cpu().numpy())
        EV, EVc, PR = (np.concatenate(EV), np.concatenate(EVc), np.concatenate(PR))
        m = inbox(G)
        for nm, ev in (("old_evidence", EV), ("old_contrast", EVc)):
            i, lift, hf = metrics(ev, m)
            res[f"{nm}_G{G}"] = {
                "box_share_of_support": float(i.mean()),
                "lift": float(np.median(lift)),
                "cells_for_half": float(hf.mean()),
                "lift_correct": float(np.median(lift[PR == labels])),
                "lift_wrong": float(np.median(lift[PR != labels])),
            }
            keep[f"{nm}_G{G}"] = (i, lift, hf)
        keep[f"oldpred_G{G}"] = PR
        if G == 20:
            EV20, EVc20 = (EV, EVc)
    data = {
        s: (
            np.load(T / f"clean_{s}.npy", mmap_mode="r"),
            np.array([r["class_id"] for r in sp[s]]),
        )
        for s in ("adapt_train", "adapt_val", "future_test")
    }
    args = types.SimpleNamespace(lr=0.001, epochs=15, batch=128)
    width = hs["weight"].shape[1]
    m10 = inbox(10)
    NEW = {}
    for arm in ("attention", "gated", "gated_shuffled"):
        model, best = sas.train(arm, 42, hs, width, classes, data, 0.0, args)
        preds, wts, _ = sas.infer(model, *data["future_test"])
        Wn = model.classifier.weight.detach()
        bn = model.classifier.bias.detach()
        tok = data["future_test"][0]
        C, maxerr = ([], 0.0)
        with torch.no_grad():
            for s in range(0, n, 128):
                h = torch.from_numpy(
                    np.asarray(tok[s : s + 128], dtype=np.float32)
                ).cuda()
                w = wts[s : s + 128].cuda()
                e = h @ Wn.T
                p = torch.from_numpy(preds[s : s + 128]).cuda().long()
                ar = torch.arange(len(p), device=h.device)
                full = (w[..., None] * h).sum(1) @ Wn.T + bn
                maxerr = max(
                    maxerr, float(((w[..., None] * e).sum(1) + bn - full).abs().max())
                )
                C.append((w * (e[ar, :, p] - e.mean(2))).cpu().numpy())
        C = np.concatenate(C)
        i, lift, hf = metrics(C, m10)
        res[f"new_{arm}"] = {
            "test_f1_check": best["val_f1"],
            "completeness_max_err": maxerr,
            "box_share_of_support": float(i.mean()),
            "lift": float(np.median(lift)),
            "cells_for_half": float(hf.mean()),
            "lift_correct": float(np.median(lift[preds == labels])),
            "lift_wrong": float(np.median(lift[preds != labels])),
        }
        NEW[arm] = (C, preds, wts.numpy())
        keep[f"new_{arm}"] = (i, lift, hf)
    i, lift, _ = keep["old_evidence_G20"]
    ok = keep["oldpred_G20"] == labels
    il, ll, _ = keep["new_gated"]
    per = {}
    for c in range(classes):
        s = ok & (labels == c)
        if s.sum() >= 15:
            per[names[c]] = {
                "n": int(s.sum()),
                "old_lift": float(np.median(lift[s])),
                "old_box_share": float(i[s].mean()),
                "new_gated_lift": float(np.median(ll[s])),
                "box_area": float(inbox(20)[s].mean()),
            }
    res["per_class"] = dict(sorted(per.items(), key=lambda kv: -kv[1]["old_lift"]))
    area = inbox(20).mean(1)
    cand = np.where(ok & (area > 0.05) & (area < 0.4) & (NEW["gated"][1] == labels))[0]
    order = cand[np.argsort(-lift[cand])]

    def distinct(idx, k):
        out, seen = ([], set())
        for j in idx:
            if labels[j] not in seen:
                out.append(j)
                seen.add(labels[j])
            if len(out) == k:
                break
        return out

    good, bad = (distinct(order, 6), distinct(order[::-1], 6))
    old8 = [893, 5931, 859, 4517, 3956, 3229, 1124, 3529]
    sel = good + bad + old8
    cache = np.load(DEFAULT_CACHE, mmap_mode="r")
    np.savez_compressed(
        OUT / "explain_examples.npz",
        idx=np.array(sel),
        group=np.array(["good"] * 6 + ["bad"] * 6 + ["earlier"] * 8),
        images=np.stack([cache[test[j]["idx"]] for j in sel]),
        boxes=boxes[sel],
        labels=labels[sel],
        old=keep["oldpred_G20"][sel],
        old_ev20=EV20[sel].astype(np.float32),
        old_ct20=EVc20[sel].astype(np.float32),
        lift=lift[sel],
        **{f"new_{a}_contrib": NEW[a][0][sel].astype(np.float32) for a in NEW},
        **{f"new_{a}_pred": NEW[a][1][sel] for a in NEW},
        **{f"new_{a}_w": NEW[a][2][sel] for a in NEW},
        names=np.array([names[c] for c in range(classes)]),
    )
    json.dump(res, open(OUT / "explain_eval.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))
    pc = list(res["per_class"].items())
    print("TOP", pc[:8])
    print("BOTTOM", pc[-8:])


def patches():
    R = Path(os.environ["FYP_RUNS"])
    T = Path(os.environ["FYP_TOKENS"])
    OUT = R / "analysis"
    rows, classes = load_rows(DEFAULT_MANIFEST)
    sp = study_splits(rows)
    n = len(sp["future_test"])
    labels = np.array([r["class_id"] for r in sp["future_test"]])
    boxes = np.array(
        [[r["box_cx"], r["box_cy"], r["box_w"], r["box_h"]] for r in sp["future_test"]]
    )
    sel = np.load(OUT / "explain_examples.npz")["idx"]
    hs = torch.load(R / "main/head.pt", weights_only=True)
    data = {
        s: (
            np.load(T / f"clean_{s}.npy", mmap_mode="r"),
            np.array([r["class_id"] for r in sp[s]]),
        )
        for s in ("adapt_train", "adapt_val", "future_test")
    }
    t20 = np.load(
        "/mnt/sdd/FYP/cache/adapter_tokens_v2/clean_future_test.npy", mmap_mode="r"
    )
    args = types.SimpleNamespace(lr=0.001, epochs=15, batch=128)
    out = {}

    def inbox20():
        c = (np.arange(20) + 0.5) / 20
        X, Y = np.meshgrid(c, c)
        return np.stack(
            [
                ((np.abs(X - cx) <= w / 2) & (np.abs(Y - cy) <= h / 2)).ravel()
                for cx, cy, w, h in boxes
            ]
        )

    M = inbox20()
    stats = {}
    for arm in ("attention", "gated", "gated_shuffled"):
        model, _ = sas.train(
            arm, 42, hs, hs["weight"].shape[1], classes, data, 0.0, args
        )
        preds, wts, _ = sas.infer(model, *data["future_test"])
        Wn = model.classifier.weight.detach()
        bn = model.classifier.bias.detach()
        C20, err, top_in = ([], 0.0, [])
        with torch.no_grad():
            for s in range(0, n, 64):
                h = torch.from_numpy(
                    np.asarray(t20[s : s + 64], dtype=np.float32)
                ).cuda()
                w10 = wts[s : s + 64].cuda().reshape(-1, 10, 10)
                w20 = (
                    w10.repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(-1, 400)
                    / 4
                )
                e = h @ Wn.T
                p = torch.from_numpy(preds[s : s + 64]).cuda().long()
                ar = torch.arange(len(p), device=h.device)
                full = (w20[..., None] * e).sum(1) + bn
                ref = model.classifier(
                    (
                        wts[s : s + 64].cuda()[..., None]
                        * torch.from_numpy(
                            np.asarray(
                                data["future_test"][0][s : s + 64], dtype=np.float32
                            )
                        ).cuda()
                    ).sum(1)
                )
                err = max(err, float((full - ref).abs().max()))
                c = (w20 * (e[ar, :, p] - e.mean(2))).cpu().numpy()
                C20.append(c)
        C20 = np.concatenate(C20)
        top = C20.argmax(1)
        top_in = M[np.arange(n), top]
        pos = np.clip(C20, 0, None)
        share = (pos * M).sum(1) / (pos.sum(1) + 1e-09)
        stats[arm] = {
            "reconstruction_err_vs_10x10_head": err,
            "top_cell_in_box": float(top_in.mean()),
            "box_share": float(share.mean()),
            "top_cell_in_box_correct": float(top_in[preds == labels].mean()),
            "top_cell_in_box_wrong": float(top_in[preds != labels].mean()),
        }
        out[f"{arm}_c20"] = C20[sel].astype(np.float32)
        out[f"{arm}_w"] = wts.numpy()[sel]
        out[f"{arm}_pred"] = preds[sel]
    np.savez_compressed(OUT / "explain20.npz", idx=sel, **out)
    print(json.dumps(stats, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "step", nargs="?", choices=("maps", "patches", "all"), default="all"
    )
    step = parser.parse_args(argv).step
    if step in ("maps", "all"):
        maps()
    if step in ("patches", "all"):
        patches()


if __name__ == "__main__":
    main()
