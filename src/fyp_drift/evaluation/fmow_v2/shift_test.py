"""Moved-object test: score heads on centred crops and on crops moved so the object sits off-centre.

Example::  python -m fyp_drift.evaluation.fmow_v2.shift_test
"""

import hashlib
import json
import os
import random
import types
from pathlib import Path

import numpy as np
import torch
from fyp_drift.evaluation import saliency_attention_study as att
from fyp_drift.evaluation import soft_saliency_heads as soft
from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_MANIFEST,
    IMAGE_SIZE,
    Backbone,
    load_rows,
    log,
    scores,
    study_splits,
    write_json,
)
from fyp_drift.evaluation.saliency_adapters import SPLITS, token_cache
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def main():
    R, T = (Path(os.environ["FYP_RUNS"]), Path(os.environ["FYP_TOKENS"]))
    OUT = R / "shift_test"
    OUT.mkdir(exist_ok=True)
    FULL = Path("/mnt/sdd/FYP/cache/shift_test_v2")
    FULL.mkdir(exist_ok=True)
    G = att.GRID
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    test = splits["future_test"]
    alpha = json.loads((R / "main/calib.json").read_text())["selected_alpha"]
    sizes = json.loads(Path(DEFAULT_MANIFEST).with_name("image_sizes.json").read_text())

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
        return {
            **r,
            "idx": k,
            "crop": (nx0, ny0, nx0 + side, ny0 + side),
            "box_cx": (cx - nx0) / side,
            "box_cy": (cy - ny0) / side,
        }

    srows = [shifted_row(r, k) for k, r in enumerate(test)]
    moved = np.array([max(abs(s["box_cx"] - 0.5), abs(s["box_cy"] - 0.5)) for s in srows])
    log(
        f"shifted test: {len(srows)} images, {np.mean(moved >= 0.15):.3f} moved >= 0.15 off-centre, median offset {np.median(moved):.3f}"
    )
    cache_path = FULL / "images_u8.npy"
    if not cache_path.with_suffix(".done").exists():
        cache = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=np.uint8, shape=(len(srows), IMAGE_SIZE, IMAGE_SIZE, 3)
        )

        class D(Dataset):

            def __len__(self):
                return len(srows)

            def __getitem__(self, i):
                with Image.open(srows[i]["local_path"]) as im:
                    im = (
                        im.convert("RGB")
                        .crop(tuple((int(round(v)) for v in srows[i]["crop"])))
                        .resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
                    )
                return (i, torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()))

        for ids, ims in DataLoader(D(), batch_size=64, num_workers=3):
            cache[ids.numpy()] = ims.numpy()
        cache.flush()
        cache_path.with_suffix(".done").write_text("ok")
    cache = np.load(cache_path, mmap_mode="r")

    def pooled(cond):
        out = FULL / f"{cond}_shift_p.npy"
        if out.with_suffix(".done").exists():
            return np.load(out, mmap_mode="r")
        backbone = Backbone("facebook/dinov2-large").cuda().eval()
        full = token_cache(backbone, cache, srows, FULL / f"{cond}_shift.npy", alpha if cond == "cloud" else 0.0, 3)
        del backbone
        torch.cuda.empty_cache()
        n, p, d = full.shape
        g = int(round(p**0.5))
        y = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(n, (g // 2) ** 2, d))
        for s in range(0, n, 256):
            b = np.asarray(full[s : s + 256], dtype=np.float32).reshape(-1, g // 2, 2, g // 2, 2, d).mean((2, 4))
            y[s : s + 256] = b.reshape(len(b), -1, d).astype(np.float16)
        y.flush()
        out.with_suffix(".done").write_text("ok")
        (FULL / f"{cond}_shift.npy").unlink()
        log("pooled shifted tokens", cond)
        return np.load(out, mmap_mode="r")

    centres = (torch.arange(G) + 0.5) / G
    yy, xx = torch.meshgrid(centres, centres, indexing="ij")
    centre_prior = torch.exp(-((yy - 0.5) ** 2 + (xx - 0.5) ** 2) / (2 * 0.2**2)).flatten()
    centre_prior /= centre_prior.sum()
    ring = torch.zeros(G, G, dtype=torch.bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    ring = ring.flatten()

    def robust(weights, rs):
        boxes = torch.stack([att.box_patch_mask(r) for r in rs]).float()
        k = boxes.sum(1).long()
        ranks = weights.argsort(1, descending=True).argsort(1)
        topk = (ranks < k[:, None]).float()
        mass = (weights * boxes).sum(1)
        prior_mass = (centre_prior[None] * boxes).sum(1)
        top = weights.argmax(1)
        return {
            "pointing_game": float(boxes.gather(1, top[:, None]).mean()),
            "box_mass_over_share": float((mass / boxes.mean(1)).mean()),
            "topk_precision": float(((topk * boxes).sum(1) / k).mean()),
            "box_mass": float(mass.mean()),
            "box_mass_minus_centre_prior": float((mass - prior_mass).mean()),
            "top_patch_on_border": float(ring[top].float().mean()),
            "centre_prior_pointing": float(boxes.gather(1, centre_prior.argmax().repeat(len(rs))[:, None]).mean()),
        }

    ARMS = {
        "clean": list(att.ARMS) + ["soft_gated", "soft_prior", "soft_gated_shuffled", "soft_prior_shuffled"],
        "cloud": [
            "attention",
            "gated",
            "gated_shuffled",
            "prior",
            "prior_shuffled",
            "soft_gated",
            "soft_gated_shuffled",
        ],
    }
    args = types.SimpleNamespace(lr=0.001, epochs=15, batch=128)
    head_state = torch.load(R / "main/head.pt", weights_only=True)
    width = head_state["weight"].shape[1]
    guide = {"clean": 0.3, "cloud": 0.1}
    for cond in ("clean", "cloud"):
        data = {
            s: (np.load(T / f"{cond}_{s}.npy", mmap_mode="r"), np.array([r["class_id"] for r in splits[s]]))
            for s in SPLITS
        }
        shift_tokens = pooled(cond)
        labels = data["future_test"][1]
        for seed in (42, 43, 44):
            for arm in ARMS[cond]:
                path = OUT / f"{cond}_{arm}_s{seed}.json"
                if path.exists():
                    continue
                if arm.startswith("soft"):
                    model, best = soft.train(arm, seed, head_state, width, classes, data, args)
                else:
                    model, best = att.train(arm, seed, head_state, width, classes, data, guide[cond], args)
                res = {"arm": arm, "condition": cond, "seed": seed, "adapt_val_macro_f1": best["val_f1"]}
                for name, toks, rs in (("centred", data["future_test"][0], test), ("shifted", shift_tokens, srows)):
                    pred, w, z = att.infer(model, toks, labels)
                    res[name] = {
                        "scores": scores(labels, pred, classes),
                        "grounding": robust(w, rs),
                        "saliency_grounding": robust(z.softmax(1), rs),
                    }
                    np.savez_compressed(
                        OUT / f"{cond}_{arm}_s{seed}_{name}.npz", weights=w.numpy().astype(np.float16), predictions=pred
                    )
                write_json(path, res)
                c, s_ = (res["centred"], res["shifted"])
                log(
                    f"{cond} {arm} s{seed}: F1 {c['scores']['macro_f1']:.3f}->{s_['scores']['macro_f1']:.3f} | point {c['grounding']['pointing_game']:.2f}->{s_['grounding']['pointing_game']:.2f} | topk {c['grounding']['topk_precision']:.2f}->{s_['grounding']['topk_precision']:.2f} | border {c['grounding']['top_patch_on_border']:.2f}"
                )
    log("DONE")


if __name__ == "__main__":
    main()
