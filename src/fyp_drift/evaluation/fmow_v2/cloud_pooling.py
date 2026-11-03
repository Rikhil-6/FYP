"""Cloud-aware pooling heads (pixel cloud score gates attention) and the head-width parameter sweep.

Example::  python -m fyp_drift.evaluation.fmow_v2.cloud_pooling
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from fyp_drift.evaluation import saliency_attention_study as att
from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_CACHE,
    DEFAULT_MANIFEST,
    IMAGE_SIZE,
    MEAN,
    STD,
    cloud_mask,
    image_cache,
    image_loader,
    load_rows,
    log,
    scores,
    study_splits,
    write_json,
)
from fyp_drift.evaluation.saliency_adapters import SPLITS
from torch import nn
from torch.nn import functional as F


def main():
    R, T = (Path(os.environ["FYP_RUNS"]), Path(os.environ["FYP_TOKENS"]))
    G = att.GRID
    CELL = IMAGE_SIZE // G
    OUT = R / "cloud_param"
    OUT.mkdir(exist_ok=True)
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    cache = image_cache(rows, DEFAULT_CACHE, 3)
    alpha = json.loads((R / "main/calib.json").read_text())["selected_alpha"]
    head_state = torch.load(R / "main/head.pt", weights_only=True)
    width = head_state["weight"].shape[1]

    def cloud_scores(cond, split):
        path = T / f"cloudscore_{cond}_{split}.npy"
        if path.exists():
            return np.load(path)
        out = []
        for images, _, _ in image_loader(cache, splits[split], 64, 3, alpha if cond == "cloud" else 0.0):
            px = (images * STD + MEAN).clamp(0, 1)
            cells = px.reshape(len(px), 3, G, CELL, G, CELL)
            white = cells.min(1).values.mean((2, 4))
            flat = 1 - (cells.mean(1).std((2, 4)) * 4).clamp(0, 1)
            out.append((white * flat).reshape(len(px), -1).half().numpy())
        out = np.concatenate(out)
        np.save(path, out)
        log("cloud scores", cond, split, out.shape)
        return out

    def detector_quality():
        c = cloud_scores("cloud", "future_test").astype(np.float32)
        truth = np.stack(
            [
                (F.avg_pool2d(cloud_mask(r["local_path"])[None], CELL)[0, 0] > 0.5).flatten().numpy()
                for r in splits["future_test"]
            ]
        )
        from sklearn.metrics import roc_auc_score

        clean = cloud_scores("clean", "future_test").astype(np.float32)
        res = {
            "cell_auc_cloud_vs_not": float(roc_auc_score(truth.flatten(), c.flatten())),
            "true_cloud_share": float(truth.mean()),
            "mean_score_cloud_cells": float(c[truth].mean()),
            "mean_score_other_cells": float(c[~truth].mean()),
            "mean_score_clean_images": float(clean.mean()),
        }
        write_json(OUT / "detector.json", res)
        log("detector", res)

    class Head(att.SaliencyAttentionHead):

        def __init__(self, kind, head_state, width, classes, hidden=128):
            base = {"cloud_gate": "attention", "cloud_gate_shuffled": "attention", "cloud_saliency_gate": "gated"}.get(
                kind, kind
            )
            super().__init__(base, head_state, width, classes)
            self.mode = kind
            self.hidden = hidden
            inputs = width
            self.scorer = (
                nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, 1))
                if hidden > 0
                else None
            )
            if kind.startswith("cloud"):
                self.kappa, self.tau = (nn.Parameter(torch.tensor(5.0)), nn.Parameter(torch.tensor(0.5)))

        def full_weights(self, tokens, z, c, generator):
            logits = (
                self.scorer(tokens).squeeze(-1)
                if self.scorer is not None
                else torch.zeros(tokens.shape[:2], device=tokens.device)
            )
            w = logits.softmax(1)
            if self.kind == "gated":
                w = w * torch.sigmoid(self.gamma * z + self.delta)
            if self.mode.startswith("cloud"):
                if self.mode.endswith("shuffled"):
                    c = c.gather(1, torch.rand(c.shape, generator=generator).argsort(1).to(c.device))
                w = w * torch.sigmoid(self.kappa * (self.tau - c))
            return w / w.sum(1, keepdim=True).clamp_min(1e-08)

        def run(self, tokens, c, generator):
            z = self.saliency(tokens, generator)
            w = self.full_weights(tokens, z, c, generator)
            return (self.classify(tokens, w), w)

    def batches(tokens, cs, labels, bs, seed=None):
        order = np.random.default_rng(seed).permutation(len(labels)) if seed is not None else np.arange(len(labels))
        for s in range(0, len(order), bs):
            idx = np.sort(order[s : s + bs])
            yield (
                torch.from_numpy(np.asarray(tokens[idx])).cuda().float(),
                torch.from_numpy(cs[idx].astype(np.float32)).cuda(),
                torch.from_numpy(labels[idx]).cuda(),
            )

    @torch.no_grad()
    def infer(model, tokens, cs, labels):
        model.eval()
        g = torch.Generator().manual_seed(7)
        P, W = ([], [])
        for x, c, _ in batches(tokens, cs, labels, 256):
            lo, w = model.run(x, c, g)
            P.append(lo.argmax(1).cpu())
            W.append(w.cpu())
        return (torch.cat(P).numpy(), torch.cat(W))

    def train_eval(kind, hidden, seed, data, cond):
        torch.manual_seed(seed)
        model = Head(kind, head_state, width, classes, hidden).cuda()
        params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum((p.numel() for p in params))
        opt = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0001)
        best = (-1, None, 0)
        for epoch in range(15):
            model.train()
            g = torch.Generator().manual_seed(seed * 1000 + epoch)
            for x, c, y in batches(*data["adapt_train"], 128, seed * 1000 + epoch):
                lo, _ = model.run(x, c, g)
                loss = F.cross_entropy(lo, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            v = scores(data["adapt_val"][2], infer(model, *data["adapt_val"])[0], classes)["macro_f1"]
            if v > best[0]:
                best = (v, {k: t.clone() for k, t in model.state_dict().items()}, epoch + 1)
        model.load_state_dict(best[1])
        pred, w = infer(model, *data["future_test"])
        labels = data["future_test"][2]
        res = {
            "kind": kind,
            "hidden": hidden,
            "seed": seed,
            "condition": cond,
            "trainable_parameters": n_params,
            "adapt_val_macro_f1": best[0],
            "best_epoch": best[2],
            "future_test": scores(labels, pred, classes),
            "predictions": pred.tolist(),
            "grounding": att.grounding(w, splits["future_test"], cond == "cloud"),
            "scalars": {n: float(getattr(model, n)) for n in ("kappa", "tau", "gamma", "delta") if hasattr(model, n)},
        }
        if cond == "cloud":
            cs = torch.from_numpy(data["future_test"][1].astype(np.float32))
            res["weight_on_high_cloud_score_cells"] = float((w * (cs > 0.5)).sum(1).mean())
        return res

    def data_for(cond):
        return {
            s: (
                np.load(T / f"{cond}_{s}.npy", mmap_mode="r"),
                cloud_scores(cond, s),
                np.array([r["class_id"] for r in splits[s]]),
            )
            for s in SPLITS
        }

    detector_quality()
    jobs = [("A", k, 128) for k in ("cloud_gate", "cloud_gate_shuffled", "cloud_saliency_gate")] + [
        ("B", k, h) for h in (0, 8, 32) for k in ("attention", "gated", "gated_shuffled")
    ]
    for cond in ("cloud", "clean"):
        data = data_for(cond)
        for part, kind, hidden in jobs:
            for seed in (42, 43, 44):
                path = OUT / f"{part}_{cond}_{kind}_h{hidden}_s{seed}.json"
                if path.exists():
                    continue
                r = train_eval(kind, hidden, seed, data, cond)
                write_json(path, r)
                log(
                    f"{part} {cond} {kind} h{hidden} s{seed}: params={r['trainable_parameters']} F1={r['future_test']['macro_f1']:.4f} acc={r['future_test']['accuracy']:.4f} focus={r['grounding']['box_mass_over_share']:.2f} "
                    + (
                        f"cloudmass={r['grounding'].get('cloud_mass', 0):.3f}/{r['grounding'].get('cloud_share', 0):.3f} "
                        if cond == "cloud"
                        else ""
                    )
                    + str(r["scalars"])
                )
    log("DONE")


if __name__ == "__main__":
    main()
