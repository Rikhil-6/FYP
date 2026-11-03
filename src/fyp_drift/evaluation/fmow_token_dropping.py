"""Test-time evidence-guided token dropping on the frozen DINOv2-L fMoW model.

For each image: predict once, rank patch tokens by their evidence for that prediction,
remove the lowest-evidence fraction right after patch embedding (so they never enter
attention), and predict again. Controls drop the same number of random patches, or the
brightest patches (a saliency-free cloud detector). No training; future-test labels are
used only for scoring.

Conditions: clean images, realistic cumulus clouds off / on the labelled object box, and
real acquisition strata (recorded cloud cover, off-nadir angle, sun elevation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_CACHE, DEFAULT_MANIFEST, Backbone, cluster_bootstrap, image_cache, image_loader, load_rows, log,
    scores, study_splits, write_json, _block_output,
)

DROP_MODES = ("evidence", "random", "brightest")


def encode_tokens(backbone, hidden):
    for block in backbone.frozen_blocks + backbone.adapted_blocks:
        hidden = _block_output(block, hidden)
    return backbone.model.layernorm(hidden)


def patch_brightness(images, patch=14):
    """Mean brightness of each patch in normalized-image units (higher = whiter)."""
    pooled = torch.nn.functional.avg_pool2d(images.mean(1, keepdim=True), patch)
    return pooled.flatten(1)


def tokens_to_keep(mode, evidence, images, keep, generator):
    if mode == "evidence":
        return evidence.topk(keep, dim=1).indices
    if mode == "brightest":
        return (-patch_brightness(images)).topk(keep, dim=1).indices
    order = torch.rand(evidence.shape, generator=generator).argsort(1)
    return order[:, :keep].to(evidence.device)


@torch.no_grad()
def predictions_with_dropping(backbone, head, data, fractions):
    """Predictions for the full model and for every (mode, fraction) of dropped tokens."""
    generator = torch.Generator().manual_seed(42)
    outputs = {"full": []} | {f"{m}_{f}": [] for m in DROP_MODES for f in fractions}
    labels = []
    for images, targets, _ in data:
        images = images.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            embedded = backbone.model.embeddings(images)
            tokens = encode_tokens(backbone, embedded)[:, 1:].float()
            logits = head(tokens.mean(1))
            evidence = (tokens * head.weight[logits.argmax(1)][:, None, :]).sum(-1)
            outputs["full"].append(logits.argmax(1).cpu())
            patches = embedded.shape[1] - 1
            for fraction in fractions:
                keep = patches - round(fraction * patches)
                for mode in DROP_MODES:
                    kept = tokens_to_keep(mode, evidence, images, keep, generator)
                    selected = torch.cat([embedded[:, :1], embedded[:, 1:].gather(1, kept[..., None].expand(-1, -1, embedded.shape[-1]))], 1)
                    pruned = encode_tokens(backbone, selected)[:, 1:].float()
                    outputs[f"{mode}_{fraction}"].append(head(pruned.mean(1)).argmax(1).cpu())
        labels.append(targets)
    return torch.cat(labels).numpy(), {k: torch.cat(v).numpy() for k, v in outputs.items()}


def real_strata(rows):
    """Masks over future-test rows for recorded acquisition nuisances (thresholds = future-test quartiles)."""
    cloud = np.array([r["cloud_cover"] for r in rows])
    nadir = np.array([r["off_nadir_angle"] for r in rows])
    sun = np.array([r["sun_elevation"] for r in rows])
    return {"all": np.ones(len(rows), bool), "cloud_cover_ge10": cloud >= 10, "cloud_free": cloud == 0,
            "off_nadir_top25": nadir >= np.quantile(nadir, 0.75), "off_nadir_bottom25": nadir <= np.quantile(nadir, 0.25),
            "low_sun_bottom25": sun <= np.quantile(sun, 0.25), "high_sun_top25": sun >= np.quantile(sun, 0.75)}


def summarize(labels, groups, outputs, strata, classes):
    table = {}
    for name, mask in strata.items():
        table[name] = {"n": int(mask.sum())} | {k: scores(labels[mask], v[mask], classes)["macro_f1"] for k, v in outputs.items()}
    paired = []
    for key in outputs:
        if not key.startswith("evidence_"):
            continue
        fraction = key.split("_", 1)[1]
        for control in ("random", "brightest", "full"):
            other = outputs["full"] if control == "full" else outputs[f"{control}_{fraction}"]
            for name, mask in strata.items():
                difference = (outputs[key][mask] == labels[mask]).astype(float) - (other[mask] == labels[mask]).astype(float)
                paired.append({"fraction": float(fraction), "vs": control, "stratum": name,
                               "delta_accuracy": cluster_bootstrap(difference, groups[mask], repetitions=1000)})
    return table, paired


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("/mnt/sdd/FYP/runs/today/main"), help="directory holding head.pt")
    parser.add_argument("--output", type=Path, default=Path("/mnt/sdd/FYP/runs/today/token_dropping"))
    parser.add_argument("--backbone", default="facebook/dinov2-large")
    parser.add_argument("--cloud-strength", type=float, default=0.8)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25])
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
    groups = np.array([r["group"] for r in test])
    report = {}
    for condition, region in (("clean", None), ("cloud_off_object", "off_object"), ("cloud_on_object", "on_object")):
        path = args.output / f"{condition}.json"
        if path.exists():
            report[condition] = json.loads(path.read_text())
            continue
        alpha = 0.0 if region is None else args.cloud_strength
        labels, outputs = predictions_with_dropping(backbone, head, image_loader(cache, test, 64, args.workers, alpha, cloud_region=region), args.fractions)
        strata = real_strata(test) if condition == "clean" else {"all": np.ones(len(test), bool)}
        table, paired = summarize(labels, groups, outputs, strata, classes)
        report[condition] = {"cloud_strength": alpha, "macro_f1": table, "paired": paired,
                             "predictions": {k: v.tolist() for k, v in outputs.items()}}
        write_json(path, report[condition])
        log(condition, json.dumps(table["all"]))
    write_json(args.output / "summary.json", {c: {"macro_f1": r["macro_f1"], "paired": r["paired"]} for c, r in report.items()})


if __name__ == "__main__":
    main()
