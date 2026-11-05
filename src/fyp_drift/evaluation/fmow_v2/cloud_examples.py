"""Cloud versions of the example figures: heads trained on clean 2014-15 images, applied to test images carrying the training cloud overlay.

Also reports whether the saliency map still lands on the object under cloud (box share of the old model's evidence and of each head's weights).
Writes cloud_examples.npz to $FYP_RUNS/analysis, the input of figures/fmow_v2/plots/fig_cloud_examples.py.
Example::  python -m fyp_drift.evaluation.fmow_v2.cloud_examples
"""

import json
import os
import types
from pathlib import Path

import numpy as np
import torch

from fyp_drift.evaluation import saliency_attention_study as sas
from fyp_drift.evaluation.fmow_dinov2l_study import DEFAULT_CACHE, DEFAULT_MANIFEST, cloud_mask, load_rows, study_splits

ALPHA = 0.85
FIG10 = (3, 5, 1)  # positions in examples.npz used by Figure 10
FIG12 = (13, 15, 16, 17)  # positions in explain_examples.npz used by Figure 12


def main():
    R, T = Path(os.environ["FYP_RUNS"]), Path(os.environ["FYP_TOKENS"])
    OUT = R / "analysis"
    rows, classes = load_rows(DEFAULT_MANIFEST)
    sp = study_splits(rows)
    test = sp["future_test"]
    n = len(test)
    labels = np.array([r["class_id"] for r in test])
    boxes = np.array([[r["box_cx"], r["box_cy"], r["box_w"], r["box_h"]] for r in test])
    names = {r["class_id"]: r["category"] for r in rows}
    hs = torch.load(R / "main/head.pt", weights_only=True)
    W, b = hs["weight"].float().cuda(), hs["bias"].float().cuda()
    data = {s: (np.load(T / f"clean_{s}.npy", mmap_mode="r"), np.array([r["class_id"] for r in sp[s]])) for s in ("adapt_train", "adapt_val", "future_test")}
    cloud10 = np.load(T / "cloud_future_test.npy", mmap_mode="r")
    cloud20 = np.load("/mnt/sdd/FYP/cache/adapter_tokens_v2/cloud_future_test.npy", mmap_mode="r")
    clean20 = np.load("/mnt/sdd/FYP/cache/adapter_tokens_v2/clean_future_test.npy", mmap_mode="r")
    args = types.SimpleNamespace(lr=0.001, epochs=15, batch=128)

    def inbox(G):
        c = (np.arange(G) + 0.5) / G
        X, Y = np.meshgrid(c, c)
        return np.stack([((np.abs(X - cx) <= w / 2) & (np.abs(Y - cy) <= h / 2)).ravel() for cx, cy, w, h in boxes])

    def evidence(tok):
        EV, PR = [], []
        with torch.no_grad():
            for s in range(0, n, 128):
                h = torch.from_numpy(np.asarray(tok[s : s + 128], dtype=np.float32)).cuda()
                e = h @ W.T
                p = (e.mean(1) + b).argmax(1)
                ar = torch.arange(len(p), device=h.device)
                EV.append((e[ar, :, p] - e.mean(2)).cpu().numpy() / e.shape[1])
                PR.append(p.cpu().numpy())
        return np.concatenate(EV), np.concatenate(PR)

    def share(ev, mask):
        pos = np.clip(ev, 0, None)
        return float(((pos * mask).sum(1) / (pos.sum(1) + 1e-9)).mean())

    m10, m20 = inbox(10), inbox(20)
    stats = {}
    EV_clean, _ = evidence(clean20)
    EV_cloud, PR_cloud = evidence(cloud20)
    for tag, ev, pr in (("clean", EV_clean, evidence(clean20)[1]), ("cloud", EV_cloud, PR_cloud)):
        stats[f"old_{tag}"] = {"box_share_of_support": share(ev, m20), "box_area": float(m20.mean()), "old_acc": float((pr == labels).mean())}
    out = {"fig10_idx": np.array([np.load(OUT / "examples.npz", allow_pickle=True)["idx"][k] for k in FIG10])}
    sel12 = np.load(OUT / "explain_examples.npz")["idx"]
    out["fig12_idx"] = np.array([sel12[k] for k in FIG12])
    show = np.concatenate([out["fig10_idx"], out["fig12_idx"]])
    cache = np.load(DEFAULT_CACHE, mmap_mode="r")
    imgs, imgs_cloud = [], []
    for j in show:
        px = np.array(cache[test[j]["idx"]]).astype(np.float32) / 255.0
        m = cloud_mask(test[j]["local_path"], 42, px.shape[0]).numpy()[0][..., None]
        imgs.append((px * 255).astype(np.uint8))
        imgs_cloud.append(((px * (1 - ALPHA * m) + ALPHA * m) * 255).round().astype(np.uint8))
    out.update(images=np.stack(imgs), images_cloud=np.stack(imgs_cloud), boxes=boxes[show], labels=labels[show], names=np.array([names[c] for c in range(classes)]),
               old_ev20_cloud=EV_cloud[show].astype(np.float32), old_pred_cloud=PR_cloud[show], old_ev20_clean=EV_clean[show].astype(np.float32))
    for arm in ("attention", "gated", "gated_shuffled"):
        model, _ = sas.train(arm, 42, hs, hs["weight"].shape[1], classes, data, 0.0, args)
        res = {}
        for tag, tok10, tok20 in (("clean", data["future_test"][0], clean20), ("cloud", cloud10, cloud20)):
            preds, wts, _ = sas.infer(model, tok10, labels)
            w10 = wts.numpy()
            Wn, bn = model.classifier.weight.detach(), model.classifier.bias.detach()
            C20 = []
            with torch.no_grad():
                for s in range(0, n, 64):
                    h = torch.from_numpy(np.asarray(tok20[s : s + 64], dtype=np.float32)).cuda()
                    w = wts[s : s + 64].cuda().reshape(-1, 10, 10).repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(-1, 400) / 4
                    e = h @ Wn.T
                    p = torch.from_numpy(preds[s : s + 64]).cuda().long()
                    ar = torch.arange(len(p), device=h.device)
                    C20.append((w * (e[ar, :, p] - e.mean(2))).cpu().numpy())
            C20 = np.concatenate(C20)
            wn = w10 / w10.sum(1, keepdims=True)
            res[tag] = {"acc": float((preds == labels).mean()), "weight_in_box": float((wn * m10).sum(1).mean()), "support_in_box": share(C20, m20)}
            if tag == "cloud":
                out[f"{arm}_w_cloud"] = w10[show]
                out[f"{arm}_c20_cloud"] = C20[show].astype(np.float32)
                out[f"{arm}_pred_cloud"] = preds[show]
            else:
                out[f"{arm}_w_clean"] = w10[show]
        stats[arm] = res
    np.savez_compressed(OUT / "cloud_examples.npz", **out)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
