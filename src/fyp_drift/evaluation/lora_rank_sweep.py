"""Parameter-matched LoRA: does LoRA still lead when it trains as few parameters as the saliency heads?

The main LoRA arm (rank 8, last 6 DINOv2-L blocks) trains 884,736 adapter weights with the historical head
frozen. The saliency attention heads train about 197k (scorer + classifier). Here LoRA is rerun, plain
cross-entropy only, at rank 2 (221,184) and rank 1 (110,592), bracketing that budget, with everything else
identical to the main arm: same historical head, cloud alpha, epochs, learning rate and seeds.

Each run is compared on the same 2016-17 test images, image by image, with the saliency heads from
saliency_attention_study (same seed and condition), using a location-clustered bootstrap of the accuracy gap.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fyp_drift.evaluation import fmow_dinov2l_study as study

COMPARE_ARMS = ("attention", "guided", "input", "gated", "prior")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("/mnt/sdd/FYP/runs/today/main"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/sdd/FYP/runs/today/lora_rank_sweep"))
    parser.add_argument("--attention", type=Path, default=Path("/mnt/sdd/FYP/runs/today/saliency_attention"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    options = parser.parse_args(argv)
    options.output.mkdir(parents=True, exist_ok=True)
    rows, classes = study.load_rows(study.DEFAULT_MANIFEST)
    splits = study.study_splits(rows)
    cache = study.image_cache(rows, study.DEFAULT_CACHE, 4)
    head_state = torch.load(options.run / "head.pt", weights_only=True)
    head = nn.Linear(head_state["weight"].shape[1], classes).cuda().eval()
    head.load_state_dict(head_state)
    head.requires_grad_(False)
    alphas = {"clean": 0.0, "cloud": json.loads((options.run / "calib.json").read_text())["selected_alpha"]}
    for rank in options.ranks:
        args = study.parse_args(["--output", str(options.run), "--rank", str(rank)])
        for seed in options.seeds:
            for condition, alpha in alphas.items():
                path = options.output / f"{condition}_plain_r{rank}_s{seed}.json"
                if path.exists():
                    continue
                started = time.time()
                result = study.train_arm(args, cache, splits, head, classes, "plain", 0.0, alpha, seed)
                study.write_json(path, {**result, "rank": rank, "condition": condition, "seconds": time.time() - started})
                study.log(f"done {path.name}: params={result['trainable_parameters']} test_f1={result['future_test']['macro_f1']:.4f}")
    summarize(options, splits)


def summarize(options, splits):
    labels = np.array([r["class_id"] for r in splits["future_test"]])
    groups = np.array([r["group"] for r in splits["future_test"]])
    lora = [json.loads(p.read_text()) for p in sorted(options.output.glob("*_plain_r*_s*.json"))]
    main_arms = {(json.loads(p.read_text())["condition"], json.loads(p.read_text())["seed"]): json.loads(p.read_text())
                 for p in sorted((options.run / "arms").glob("*_plain_l0.0_s*.json"))}
    for (condition, seed), r in main_arms.items():
        lora.append({**r, "rank": 8, "condition": condition, "seed": seed})
    table, paired = {}, []
    for rank in sorted({r["rank"] for r in lora}):
        for condition in ("clean", "cloud"):
            rs = [r for r in lora if (r["rank"], r["condition"]) == (rank, condition)]
            if not rs:
                continue
            table[f"{condition}/lora_r{rank}"] = {"seeds": len(rs), "trainable_parameters": rs[0]["trainable_parameters"],
                                                  "test_macro_f1": float(np.mean([r["future_test"]["macro_f1"] for r in rs])),
                                                  "test_accuracy": float(np.mean([r["future_test"]["accuracy"] for r in rs]))}
            for r in rs:
                for arm in COMPARE_ARMS:
                    other = options.attention / f"{condition}_{arm}_s{r['seed']}.json"
                    if not other.exists():
                        continue
                    o = json.loads(other.read_text())
                    difference = (np.array(o["predictions"]) == labels).astype(float) - (np.array(r["predictions"]) == labels).astype(float)
                    paired.append({"condition": condition, "rank": rank, "seed": r["seed"], "saliency_arm": arm,
                                   "delta_macro_f1_arm_minus_lora": o["future_test"]["macro_f1"] - r["future_test"]["macro_f1"],
                                   "delta_accuracy_arm_minus_lora": study.cluster_bootstrap(difference, groups, repetitions=1000)})
    study.write_json(options.output / "summary.json", {"table": table, "paired": paired})
    study.log("summary", json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
