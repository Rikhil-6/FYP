"""Per-patch maps for the old model, soft gate, LoRA and hybrid on the saved example images, clean and with cloud.

Run from the repo folder that contains models/fmow_v2 (package on the path, GPU optional):
    PYTHONPATH=src python lora_examples.py
    PYTHONPATH=src python lora_examples.py --hybrid models/fmow_v2/hybrid --lora models/fmow_v2/lora_r2
Writes lora_examples.npz next to the input; send that file back.
Map meaning: old and lora = standardised evidence of the linear head per 10 x 10 cell (its saliency);
soft_gate and hybrid = the head's learned weight per cell.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from fyp_drift.evaluation.fmow_dinov2l_study import MEAN, STD
from fyp_drift.evaluation.fmow_v2.context import pool_2x2
from fyp_drift.evaluation.fmow_v2.models import GRID, Predictor


def to_input(images, device):
    out = [((torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1) - MEAN) / STD) for im in images]
    return torch.stack(out).to(device)


@torch.no_grad()
def run(kind, P, x):
    tokens = P.backbone(x)[:, 1:].float()
    cells = pool_2x2(tokens)
    if kind in ("old", "lora"):
        hist = P.historical
        logits = hist(cells.mean(1))
        pred = logits.argmax(1)
        ev = (cells * hist.weight[pred][:, None, :]).sum(-1)
        maps = (ev - ev.mean(1, keepdim=True)) / ev.std(1, keepdim=True).clamp_min(1e-6)
    elif kind == "hybrid":
        logits, maps = P.head(cells, P.saliency_reference(x))
    else:
        logits, maps, _ = P.head(cells, torch.Generator().manual_seed(7))
    prob = logits.softmax(1)
    return prob.argmax(1).cpu().numpy(), prob.max(1).values.cpu().numpy(), maps.reshape(-1, GRID, GRID).cpu().numpy()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="figures/fmow_v2/plots/data/cloud_examples.npz")
    a.add_argument("--historical", default="models/fmow_v2/historical")
    a.add_argument("--soft", default="models/fmow_v2/soft_gate")
    a.add_argument("--lora", default="models/fmow_v2/lora_r2")
    a.add_argument("--hybrid", default="models/fmow_v2/hybrid")
    args = a.parse_args()
    d = np.load(args.data, allow_pickle=True)
    out = {k: d[k] for k in ("images", "images_cloud", "boxes", "labels", "names")}
    for tag, kind, path in (("old", "old", args.historical), ("soft", "soft", args.soft), ("lora", "lora", args.lora), ("hybrid", "hybrid", args.hybrid)):
        P = Predictor.load(path)
        dev = next(P.backbone.parameters()).device
        for cond, key in (("clean", "images"), ("cloud", "images_cloud")):
            pred, conf, maps = run(kind, P, to_input(d[key], dev))
            out[f"{tag}_pred_{cond}"], out[f"{tag}_conf_{cond}"], out[f"{tag}_map_{cond}"] = pred, conf, maps
        print(tag, "kind", P.config["kind"], "rank", P.config.get("rank"), "blocks", P.config.get("blocks"))
        del P
        torch.cuda.empty_cache()
    dest = Path(args.data).with_name("lora_examples.npz")
    np.savez_compressed(dest, **out)
    print("saved", dest)


if __name__ == "__main__":
    main()
