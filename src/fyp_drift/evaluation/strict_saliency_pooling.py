"""Strict, training-free saliency pooling on the frozen DINOv2-L fMoW model.

Nothing is trained or tuned: the frozen backbone and the frozen historical head are used
as they are. The only change is which patch tokens the head reads. Saliency pooling keeps
the top fraction of patches by the frozen model's evidence for its own predicted class;
controls read every patch (the unmodified model) or the same number of random patches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_CACHE, DEFAULT_MANIFEST, Backbone, cluster_bootstrap, image_cache, image_loader, load_rows, log, scores,
    study_splits, write_json,
)
from fyp_drift.evaluation.fmow_token_dropping import real_strata


@torch.no_grad()
def pooled_predictions(backbone, head, data, fractions):
    """Predictions from all-patch pooling and from evidence / random top-k pooling at each fraction."""
    generator = torch.Generator().manual_seed(42)
    outputs = {"all_patches": []} | {f"{mode}_{f}": [] for mode in ("evidence", "random") for f in fractions}
    labels = []
    for images, targets, _ in data:
        with torch.autocast("cuda", dtype=torch.float16):
            tokens = backbone(images.cuda(non_blocking=True))[:, 1:].float()
        logits = head(tokens.mean(1))
        outputs["all_patches"].append(logits.argmax(1).cpu())
        evidence = (tokens * head.weight[logits.argmax(1)][:, None, :]).sum(-1)
        for fraction in fractions:
            keep = round(fraction * tokens.shape[1])
            for mode in ("evidence", "random"):
                order = evidence.topk(keep, 1).indices if mode == "evidence" else torch.rand(evidence.shape, generator=generator).argsort(1)[:, :keep].to(tokens.device)
                pooled = tokens.gather(1, order[..., None].expand(-1, -1, tokens.shape[-1])).mean(1)
                outputs[f"{mode}_{fraction}"].append(head(pooled).argmax(1).cpu())
        labels.append(targets)
    return torch.cat(labels).numpy(), {k: torch.cat(v).numpy() for k, v in outputs.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("/mnt/sdd/FYP/runs/today/main"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/sdd/FYP/runs/today/strict_saliency"))
    parser.add_argument("--backbone", default="facebook/dinov2-large")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--realistic-strength", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, classes = load_rows(DEFAULT_MANIFEST)
    test = study_splits(rows)["future_test"]
    cache = image_cache(rows, DEFAULT_CACHE, args.workers)
    head_state = torch.load(args.run / "head.pt", weights_only=True)
    head = nn.Linear(head_state["weight"].shape[1], classes).cuda().eval()
    head.load_state_dict(head_state)
    backbone = Backbone(args.backbone).cuda().eval()
    main_cloud = json.loads((args.run / "calib.json").read_text())["selected_alpha"]
    groups = np.array([r["group"] for r in test])
    conditions = {"clean": (0.0, None), "main_cloud": (main_cloud, None),
                  "realistic_cloud_off_object": (args.realistic_strength, "off_object"),
                  "realistic_cloud_on_object": (args.realistic_strength, "on_object")}
    summary = {}
    for condition, (alpha, region) in conditions.items():
        labels, outputs = pooled_predictions(backbone, head, image_loader(cache, test, 64, args.workers, alpha, cloud_region=region), args.fractions)
        strata = real_strata(test) if condition == "clean" else {"all": np.ones(len(test), bool)}
        table = {name: {"n": int(mask.sum())} | {k: scores(labels[mask], v[mask], classes)["macro_f1"] for k, v in outputs.items()} for name, mask in strata.items()}
        paired = []
        for fraction in args.fractions:
            for control in ("random", "all_patches"):
                other = outputs["all_patches"] if control == "all_patches" else outputs[f"random_{fraction}"]
                for name, mask in strata.items():
                    difference = (outputs[f"evidence_{fraction}"][mask] == labels[mask]).astype(float) - (other[mask] == labels[mask]).astype(float)
                    paired.append({"fraction": fraction, "vs": control, "stratum": name, "delta_accuracy": cluster_bootstrap(difference, groups[mask], repetitions=1000)})
        summary[condition] = {"strength": alpha, "region": region, "macro_f1": table, "paired": paired}
        write_json(args.output / "summary.json", summary)
        log(condition, json.dumps({k: round(v, 4) for k, v in table["all"].items() if k != "n"}))


if __name__ == "__main__":
    main()
