"""Split the new heads' exact 10x10 cell contributions into 20x20 patch contributions for the example images.

Example::  python -m fyp_drift.evaluation.fmow_v2.explain_patches
"""

import json
import os
import types
from pathlib import Path

import numpy as np
import torch
from fyp_drift.evaluation import saliency_attention_study as sas
from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_MANIFEST,
    load_rows,
    study_splits,
)


def main():
    R = Path(os.environ["FYP_RUNS"])
    T = Path(os.environ["FYP_TOKENS"])
    OUT = R / "analysis"
    rows, classes = load_rows(DEFAULT_MANIFEST)
    sp = study_splits(rows)
    n = len(sp["future_test"])
    labels = np.array([r["class_id"] for r in sp["future_test"]])
    boxes = np.array([[r["box_cx"], r["box_cy"], r["box_w"], r["box_h"]] for r in sp["future_test"]])
    sel = np.load(OUT / "explain_examples.npz")["idx"]
    hs = torch.load(R / "main/head.pt", weights_only=True)
    data = {
        s: (np.load(T / f"clean_{s}.npy", mmap_mode="r"), np.array([r["class_id"] for r in sp[s]]))
        for s in ("adapt_train", "adapt_val", "future_test")
    }
    t20 = np.load("/mnt/sdd/FYP/cache/adapter_tokens_v2/clean_future_test.npy", mmap_mode="r")
    args = types.SimpleNamespace(lr=0.001, epochs=15, batch=128)
    out = {}

    def inbox20():
        c = (np.arange(20) + 0.5) / 20
        X, Y = np.meshgrid(c, c)
        return np.stack([((np.abs(X - cx) <= w / 2) & (np.abs(Y - cy) <= h / 2)).ravel() for cx, cy, w, h in boxes])

    M = inbox20()
    stats = {}
    for arm in ("attention", "gated", "gated_shuffled"):
        model, _ = sas.train(arm, 42, hs, hs["weight"].shape[1], classes, data, 0.0, args)
        preds, wts, _ = sas.infer(model, *data["future_test"])
        Wn = model.classifier.weight.detach()
        bn = model.classifier.bias.detach()
        C20, err, top_in = ([], 0.0, [])
        with torch.no_grad():
            for s in range(0, n, 64):
                h = torch.from_numpy(np.asarray(t20[s : s + 64], dtype=np.float32)).cuda()
                w10 = wts[s : s + 64].cuda().reshape(-1, 10, 10)
                w20 = w10.repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(-1, 400) / 4
                e = h @ Wn.T
                p = torch.from_numpy(preds[s : s + 64]).cuda().long()
                ar = torch.arange(len(p), device=h.device)
                full = (w20[..., None] * e).sum(1) + bn
                ref = model.classifier(
                    (
                        wts[s : s + 64].cuda()[..., None]
                        * torch.from_numpy(np.asarray(data["future_test"][0][s : s + 64], dtype=np.float32)).cuda()
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


if __name__ == "__main__":
    main()
