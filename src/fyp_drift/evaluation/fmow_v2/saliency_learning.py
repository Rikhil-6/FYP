"""Saliency in the training signal only: saliency-guided training and saliency-ranked patch dropping, with random controls.

Example::  python -m fyp_drift.evaluation.fmow_v2.saliency_learning
"""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from fyp_drift.evaluation import saliency_attention_study as att
from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_MANIFEST,
    load_rows,
    log,
    scores,
    study_splits,
    write_json,
)
from fyp_drift.evaluation.saliency_adapters import SPLITS, batches
from torch.nn import functional as F


def main():
    R, T = (Path(os.environ["FYP_RUNS"]), Path(os.environ["FYP_TOKENS"]))
    SHIFT = Path("/mnt/sdd/FYP/cache/shift_test_v2")
    OUT = R / "saliency_learning"
    OUT.mkdir(exist_ok=True)
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    test = splits["future_test"]
    sizes = json.loads(Path(DEFAULT_MANIFEST).with_name("image_sizes.json").read_text())
    head_state = torch.load(R / "main/head.pt", weights_only=True)
    width = head_state["weight"].shape[1]

    def shifted_row(r, k):
        x0, y0, x1, y1 = r["crop"]
        side = x1 - x0
        cx, cy, bw, bh = (x0 + r["box_cx"] * side, y0 + r["box_cy"] * side, r["box_w"] * side, r["box_h"] * side)
        W, H = sizes[r["key"]]
        rng = random.Random(int(hashlib.sha256(r["key"].encode()).hexdigest()[:8], 16))
        bx, by = (min(0.5, bw / side / 2), min(0.5, bh / side / 2))
        best = (x0, y0)
        for _ in range(50):
            u, v = (rng.uniform(bx, 1 - bx), rng.uniform(by, 1 - by))
            if max(abs(u - 0.5), abs(v - 0.5)) < 0.15:
                continue
            nx0, ny0 = (min(max(cx - u * side, 0), W - side), min(max(cy - v * side, 0), H - side))
            if max(abs((cx - nx0) / side - 0.5), abs((cy - ny0) / side - 0.5)) >= 0.15:
                best = (nx0, ny0)
                break
        nx0, ny0 = best
        return {**r, "box_cx": (cx - nx0) / side, "box_cy": (cy - ny0) / side}

    srows = [shifted_row(r, k) for k, r in enumerate(test)]

    def keep_mask(z, mode, generator):
        n = z.shape[1]
        if mode in ("sgt", "focus_aug"):
            order, frac = (z.argsort(1), 0.5)
        elif mode == "context_aug":
            order, frac = (z.argsort(1, descending=True), 0.25)
        else:
            order, frac = (torch.rand(z.shape, generator=generator).argsort(1).to(z.device), 0.5)
        k = round(frac * n)
        return torch.ones_like(z).scatter_(1, order[:, :k], 0.0)

    def masked_logits(model, tokens, z, keep):
        w = model.weights(tokens, z) * keep
        return model.classify(tokens, w / w.sum(1, keepdim=True).clamp_min(1e-08))

    def train(mode, seed, data):
        torch.manual_seed(seed)
        model = att.SaliencyAttentionHead("attention", head_state, width, classes).cuda()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
        best = {"val_f1": -1}
        for epoch in range(15):
            model.train()
            g = torch.Generator().manual_seed(seed * 1000 + epoch)
            for x, y in batches(*data["adapt_train"], 128, shuffle_seed=seed * 1000 + epoch):
                z = model.saliency(x, g)
                keep = keep_mask(z, mode, g)
                if mode in ("sgt", "sgt_random"):
                    full = model.classify(x, model.weights(x, z))
                    masked = masked_logits(model, x, z, keep)
                    loss = (
                        F.cross_entropy(masked, y)
                        + F.kl_div(masked.log_softmax(1), full.softmax(1).detach(), reduction="batchmean")
                        + F.cross_entropy(full, y)
                    )
                else:
                    loss = F.cross_entropy(masked_logits(model, x, z, keep), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            v = scores(data["adapt_val"][1], att.infer(model, *data["adapt_val"])[0], classes)["macro_f1"]
            if v > best["val_f1"]:
                best = {"val_f1": v, "epoch": epoch + 1, "state": {k: t.clone() for k, t in model.state_dict().items()}}
        model.load_state_dict(best["state"])
        return (model.eval(), best)

    MODES = ("sgt", "sgt_random", "focus_aug", "context_aug", "random_aug")
    for cond in ("clean", "cloud"):
        data = {
            s: (np.load(T / f"{cond}_{s}.npy", mmap_mode="r"), np.array([r["class_id"] for r in splits[s]]))
            for s in SPLITS
        }
        labels = data["future_test"][1]
        for seed in (42, 43, 44):
            for mode in MODES:
                path = OUT / f"{cond}_{mode}_s{seed}.json"
                if path.exists():
                    continue
                model, best = train(mode, seed, data)
                pred, w, z = att.infer(model, *data["future_test"])
                res = {
                    "mode": mode,
                    "condition": cond,
                    "seed": seed,
                    "best_epoch": best["epoch"],
                    "adapt_val_macro_f1": best["val_f1"],
                    "future_test": scores(labels, pred, classes),
                    "predictions": pred.tolist(),
                    "faithfulness": att.faithfulness(model, *data["future_test"], w, pred),
                    "grounding": att.grounding(w, test, cond == "cloud"),
                }
                sp = SHIFT / f"{cond}_shift_p.npy"
                if sp.with_suffix(".done").exists():
                    sp_pred, sw, _ = att.infer(model, np.load(sp, mmap_mode="r"), labels)
                    res["shifted"] = {
                        "scores": scores(labels, sp_pred, classes),
                        "grounding": att.grounding(sw, srows, False),
                    }
                write_json(path, res)
                log(
                    f"{cond} {mode} s{seed}: F1={res['future_test']['macro_f1']:.4f} focus={res['grounding']['box_mass_over_share']:.2f} point={res['grounding']['pointing_game']:.2f} top25del={res['future_test']['accuracy'] - res['faithfulness']['delete_top_0.25']['accuracy']:.3f}"
                    + (
                        f" | shifted F1={res['shifted']['scores']['macro_f1']:.4f} point={res['shifted']['grounding']['pointing_game']:.2f}"
                        if "shifted" in res
                        else ""
                    )
                )
    log("DONE")


if __name__ == "__main__":
    main()
