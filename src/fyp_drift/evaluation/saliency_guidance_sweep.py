"""Guidance-strength sweep for the saliency-guided attention adapter (frozen DINOv2-L).

Reuses the cached frozen tokens from fyp_drift.evaluation.saliency_adapters. For each
condition the guidance weight is chosen by mean adapt-val macro-F1 across seeds; future-test
labels are read only to score the chosen weight against the unguided attention adapter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fyp_drift.evaluation.fmow_dinov2l_study import DEFAULT_MANIFEST, cluster_bootstrap, load_rows, log, study_splits, write_json
from fyp_drift.evaluation.saliency_adapters import SPLITS, train_arm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("/mnt/sdd/FYP/runs/today/main"))
    parser.add_argument("--adapters", type=Path, default=Path("/mnt/sdd/FYP/runs/today/saliency_adapters"))
    parser.add_argument("--tokens", type=Path, default=Path("/mnt/sdd/FYP/cache/adapter_tokens"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/sdd/FYP/runs/today/guidance_sweep"))
    parser.add_argument("--weights", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    labels = np.array([r["class_id"] for r in splits["future_test"]])
    groups = np.array([r["group"] for r in splits["future_test"]])
    head_state = torch.load(args.run / "head.pt", weights_only=True)
    width = head_state["weight"].shape[1]
    report = {}
    for condition in ("clean", "cloud"):
        data = {s: (np.load(args.tokens / f"{condition}_{s}.npy", mmap_mode="r"), np.array([r["class_id"] for r in splits[s]])) for s in SPLITS}
        runs = {1.0: [json.loads((args.adapters / f"{condition}_guided_s{seed}.json").read_text()) for seed in args.seeds]}
        for weight in args.weights:
            runs[weight] = []
            for seed in args.seeds:
                path = args.output / f"{condition}_guided_w{weight}_s{seed}.json"
                if not path.exists():
                    args.guide_weight = weight
                    write_json(path, {**train_arm("guided", seed, head_state, width, classes, data, args), "condition": condition, "guide_weight": weight})
                runs[weight].append(json.loads(path.read_text()))
                log(condition, weight, seed, round(runs[weight][-1]["future_test"]["macro_f1"], 4))
        attention = [json.loads((args.adapters / f"{condition}_attention_s{seed}.json").read_text()) for seed in args.seeds]
        by_weight = {w: {"adapt_val": float(np.mean([r["adapt_val_macro_f1"] for r in rs])), "test": float(np.mean([r["future_test"]["macro_f1"] for r in rs]))} for w, rs in runs.items()}
        chosen = max(by_weight, key=lambda w: by_weight[w]["adapt_val"])
        paired = []
        for guided, plain in zip(runs[chosen], attention):
            difference = (np.array(guided["predictions"]) == labels).astype(float) - (np.array(plain["predictions"]) == labels).astype(float)
            paired.append({"seed": guided["seed"], "delta_macro_f1": guided["future_test"]["macro_f1"] - plain["future_test"]["macro_f1"],
                           "delta_accuracy": cluster_bootstrap(difference, groups, repetitions=1000)})
        report[condition] = {"by_weight": {str(w): v for w, v in by_weight.items()}, "chosen_weight": chosen,
                             "attention_test_mean": float(np.mean([r["future_test"]["macro_f1"] for r in attention])), "chosen_vs_attention": paired}
        write_json(args.output / "summary.json", report)
    log("sweep summary", json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
