"""Extra evaluation metrics beyond accuracy and macro-F1.

  saved  (CPU, from predictions already on disk)
         balanced accuracy, worst-class F1, bottom-quartile class F1, worst-region accuracy (WILDS-style
         continents), clean-to-cloud retention, McNemar tests against learned attention and LoRA rank 8,
         router risk-coverage, route regret and the either-model-correct oracle.
  heads  (GPU, re-trains the attention heads on cached tokens and keeps their outputs)
         expected calibration error, Brier score, negative log-likelihood; box IoU of the top-weighted
         patches; deletion and insertion curves with their areas. Probabilities and patch weights are saved
         per run (npz) for figures and later analysis.

Class-level IoU (Jaccard) is not reported separately: per class it equals F1 / (2 - F1), so it carries no
information beyond macro-F1. Spatial IoU here means attention-vs-object-box overlap.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score

from fyp_drift.evaluation.fmow_dinov2l_study import DEFAULT_MANIFEST, load_rows, log, scores, study_splits, write_json
from fyp_drift.evaluation.saliency_adapters import SPLITS, batches
from fyp_drift.evaluation.saliency_attention_study import ARMS, box_patch_mask, train

import os
RUNS = Path(os.environ.get("FYP_RUNS", os.environ.get("FYP_ROOT", "/mnt/sdd/FYP") + "/runs/today"))
TOKENS = Path(os.environ.get("FYP_TOKENS", os.environ.get("FYP_ROOT", "/mnt/sdd/FYP") + "/cache/adapter_tokens"))
SEEDS = (42, 43, 44)
CONTINENT_TO_REGION = {"NA": "Americas", "SA": "Americas", "EU": "Europe", "AS": "Asia", "AF": "Africa", "OC": "Oceania"}
GUIDE = {"clean": 0.3, "cloud": 0.1}


def region_of(country):
    try:
        import pycountry_convert as pc
        alpha2 = pc.country_alpha3_to_country_alpha2(country) if len(country) == 3 else country
        return CONTINENT_TO_REGION.get(pc.country_alpha2_to_continent_code(alpha2), "Other")
    except Exception:
        return "Other"


def classification_metrics(labels, predictions, regions):
    present = np.unique(labels)
    per_class = f1_score(labels, predictions, labels=present, average=None, zero_division=0)
    supported = np.array([(labels == c).sum() >= 5 for c in present])
    region_accuracy = {r: float((predictions[regions == r] == labels[regions == r]).mean()) for r in np.unique(regions) if (regions == r).sum() >= 30}
    return {"accuracy": float((predictions == labels).mean()), "macro_f1": float(per_class.mean()),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "worst_class_f1": float(per_class[supported].min()), "bottom_quartile_class_f1": float(np.sort(per_class)[: max(1, len(per_class) // 4)].mean()),
            "region_accuracy": region_accuracy, "worst_region": min(region_accuracy, key=region_accuracy.get),
            "worst_region_accuracy": min(region_accuracy.values())}


def mcnemar(correct_a, correct_b):
    only_a, only_b = int((correct_a & ~correct_b).sum()), int((~correct_a & correct_b).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}


def prediction_sources():
    """(method, condition, seed) -> predictions array, for every run saved today."""
    sources = {}
    for condition in ("clean", "cloud"):
        frozen = RUNS / "main/arms" / f"frozen_{condition}.json"
        if frozen.exists():
            for seed in SEEDS:
                sources[("frozen", condition, seed)] = np.array(json.loads(frozen.read_text())["predictions"])
        for seed in SEEDS:
            for mode, lam in (("plain", 0.0), ("uniform", 1.0), ("shuffled", 1.0), ("saliency", 1.0)):
                path = RUNS / "main/arms" / f"{condition}_{mode}_l{lam}_s{seed}.json"
                if path.exists():
                    sources[(f"lora_r8_{mode}", condition, seed)] = np.array(json.loads(path.read_text())["predictions"])
            for rank in (2, 1):
                path = RUNS / "lora_rank_sweep" / f"{condition}_plain_r{rank}_s{seed}.json"
                if path.exists():
                    sources[(f"lora_r{rank}_plain", condition, seed)] = np.array(json.loads(path.read_text())["predictions"])
            for arm in ARMS:
                path = RUNS / "saliency_attention" / f"{condition}_{arm}_s{seed}.json"
                if path.exists():
                    sources[(f"head_{arm}", condition, seed)] = np.array(json.loads(path.read_text())["predictions"])
            for arm in ("input", "gated"):
                path = RUNS / "saliency_router" / f"{condition}_{arm}_s{seed}_route_log.csv"
                if path.exists():
                    with open(path) as handle:
                        sources[(f"router_{arm}", condition, seed)] = np.array([int(r["final_pred"]) for r in csv.DictReader(handle)])
    return sources


def risk_coverage(path, labels):
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    old, new = np.array([int(r["historical_pred"]) for r in rows]), np.array([int(r["updated_pred"]) for r in rows])
    p, routed = np.array([float(r["p_update"]) for r in rows]), np.array([int(r["routed"]) for r in rows])
    disagree = np.flatnonzero(old != new)
    order = disagree[np.argsort(-p[disagree])]
    curve = []
    for fraction in np.linspace(0, 1, 11):
        chosen = order[: round(fraction * len(order))]
        final = old.copy()
        final[chosen] = new[chosen]
        curve.append({"coverage": float(len(chosen) / len(labels)), "accuracy": float((final == labels).mean())})
    final = np.where(routed == 1, new, old)
    return {"curve": curve, "area_accuracy_over_disagreement_fraction": float(np.mean([c["accuracy"] for c in curve])),
            "historical_accuracy": float((old == labels).mean()), "updated_accuracy": float((new == labels).mean()),
            "routed_accuracy": float((final == labels).mean()), "route_regret_points": float(100 * ((new == labels).mean() - (final == labels).mean())),
            "oracle_accuracy": float(((old == labels) | (new == labels)).mean())}


def run_saved(output):
    rows, _ = load_rows(DEFAULT_MANIFEST)
    test = study_splits(rows)["future_test"]
    labels = np.array([r["class_id"] for r in test])
    regions = np.array([region_of(r["country"]) for r in test])
    log("regions", {r: int((regions == r).sum()) for r in np.unique(regions)}, "sample country codes", sorted({r["country"] for r in test})[:8])
    sources = prediction_sources()
    per_run = {f"{m}/{c}/s{s}": classification_metrics(labels, p, regions) for (m, c, s), p in sources.items()}
    table = {}
    for method in sorted({m for m, _, _ in sources}):
        for condition in ("clean", "cloud"):
            runs = [per_run[f"{method}/{condition}/s{s}"] for s in SEEDS if f"{method}/{condition}/s{s}" in per_run]
            if runs:
                table[f"{method}/{condition}"] = {k: float(np.mean([r[k] for r in runs])) for k in ("accuracy", "macro_f1", "balanced_accuracy", "worst_class_f1", "bottom_quartile_class_f1", "worst_region_accuracy")} | {"seeds": len(runs), "worst_region": runs[0]["worst_region"]}
    retention = {}
    for method in sorted({m for m, _, _ in sources}):
        drops = [per_run[f"{method}/clean/s{s}"]["macro_f1"] - per_run[f"{method}/cloud/s{s}"]["macro_f1"] for s in SEEDS
                 if f"{method}/clean/s{s}" in per_run and f"{method}/cloud/s{s}" in per_run]
        if drops:
            retention[method] = {"clean_minus_cloud_macro_f1_points": float(100 * np.mean(drops)), "seeds": len(drops)}
    tests = []
    for (method, condition, seed), predictions in sources.items():
        for reference in ("head_attention", "lora_r8_plain"):
            other = sources.get((reference, condition, seed))
            if other is not None and method != reference:
                tests.append({"method": method, "reference": reference, "condition": condition, "seed": seed, **mcnemar(predictions == labels, other == labels)})
    routers = {}
    for path in sorted((RUNS / "saliency_router").glob("*_route_log.csv")):
        routers[path.name.replace("_route_log.csv", "")] = risk_coverage(path, labels)
    write_json(output / "saved_metrics.json", {"table": table, "retention": retention, "mcnemar": tests, "router": routers, "per_run": per_run})
    for k, v in table.items():
        log(k, {m: round(x, 4) if isinstance(x, float) else x for m, x in v.items()})
    log("retention", json.dumps(retention))


def calibration(probabilities, labels, bins=15):
    confidence, predicted = probabilities.max(1), probabilities.argmax(1)
    correct = (predicted == labels).astype(float)
    edges, ece = np.linspace(0, 1, bins + 1), 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            ece += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    onehot = np.eye(probabilities.shape[1])[labels]
    return {"ece": float(ece), "brier": float(((probabilities - onehot) ** 2).sum(1).mean()),
            "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean())}


@torch.no_grad()
def head_outputs(model, tokens, labels, box_masks):
    model.eval()
    generator = torch.Generator().manual_seed(7)
    probabilities, weights = [], []
    deletion, insertion = defaultdict(list), defaultdict(list)
    fractions_delete, fractions_insert = np.round(np.arange(0.0, 1.0, 0.1), 1), np.round(np.arange(0.1, 1.01, 0.1), 1)
    for x, y in batches(tokens, labels, 256):
        logits, w, _ = model(x, generator)
        probabilities.append(logits.softmax(1).cpu())
        weights.append(w.cpu())
        order = w.argsort(1, descending=True)
        for f in fractions_delete:
            keep = torch.ones_like(w).scatter_(1, order[:, : round(f * w.shape[1])], 0.0)
            kept = w * keep
            deletion[f].append((model.classify(x, kept / kept.sum(1, keepdim=True).clamp_min(1e-8)).argmax(1) == y).cpu())
        for f in fractions_insert:
            keep = torch.zeros_like(w).scatter_(1, order[:, : round(f * w.shape[1])], 1.0)
            kept = w * keep
            insertion[f].append((model.classify(x, kept / kept.sum(1, keepdim=True).clamp_min(1e-8)).argmax(1) == y).cpu())
    probabilities, weights = torch.cat(probabilities).numpy(), torch.cat(weights)
    deletion = {float(f): float(torch.cat(v).float().mean()) for f, v in deletion.items()}
    insertion = {float(f): float(torch.cat(v).float().mean()) for f, v in insertion.items()}
    boxes = torch.stack(box_masks)
    k_box = boxes.sum(1).long()
    order = weights.argsort(1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(weights.shape[1]).expand_as(order).contiguous())
    top_box = ranks < k_box[:, None]
    top_quarter = ranks < weights.shape[1] // 4
    iou = lambda top: ((top & boxes).sum(1).float() / (top | boxes).sum(1).float()).mean().item()
    return probabilities, weights, {"deletion_curve": deletion, "insertion_curve": insertion,
                                    "deletion_area": float(np.mean(list(deletion.values()))), "insertion_area": float(np.mean(list(insertion.values()))),
                                    "box_iou_top_k_equal_box": iou(top_box), "box_iou_top_quarter": iou(top_quarter)}


def run_heads(output, epochs):
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    test = splits["future_test"]
    labels = np.array([r["class_id"] for r in test])
    box_masks = [box_patch_mask(r) for r in test]
    head_state = torch.load(RUNS / "main/head.pt", weights_only=True)
    width = head_state["weight"].shape[1]
    train_args = SimpleNamespace(epochs=epochs, batch=128, lr=1e-3)
    heads_dir = output / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    for condition in ("clean", "cloud"):
        data = {s: (np.load(TOKENS / f"{condition}_{s}.npy", mmap_mode="r"), np.array([r["class_id"] for r in splits[s]])) for s in SPLITS}
        for seed in SEEDS:
            for arm in ARMS:
                stem = f"{condition}_{arm}_s{seed}"
                if (heads_dir / f"{stem}.json").exists():
                    continue
                model, best = train(arm, seed, head_state, width, classes, data, GUIDE[condition], train_args)
                probabilities, weights, extra = head_outputs(model, *data["future_test"], box_masks)
                predictions = probabilities.argmax(1)
                np.savez_compressed(heads_dir / f"{stem}.npz", probabilities=probabilities.astype(np.float16), weights=weights.numpy().astype(np.float16))
                result = {"arm": arm, "condition": condition, "seed": seed, "best_epoch": best["epoch"],
                          "future_test": scores(labels, predictions, classes), **calibration(probabilities.astype(np.float64), labels), **extra}
                write_json(heads_dir / f"{stem}.json", result)
                log(f"{stem}: f1={result['future_test']['macro_f1']:.4f} ece={result['ece']:.3f} iou_k={result['box_iou_top_k_equal_box']:.3f} "
                    f"del_area={result['deletion_area']:.3f} ins_area={result['insertion_area']:.3f}")
    runs = [json.loads(p.read_text()) for p in sorted(heads_dir.glob("*.json"))]
    table = {}
    for condition in ("clean", "cloud"):
        for arm in ARMS:
            rs = [r for r in runs if (r["condition"], r["arm"]) == (condition, arm)]
            if rs:
                table[f"{condition}/{arm}"] = {k: float(np.mean([r[k] for r in rs])) for k in ("ece", "brier", "nll", "box_iou_top_k_equal_box", "box_iou_top_quarter", "deletion_area", "insertion_area")} | {
                    "macro_f1": float(np.mean([r["future_test"]["macro_f1"] for r in rs])), "seeds": len(rs)}
    write_json(output / "head_metrics.json", table)
    log("head summary", json.dumps(table, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=("saved", "heads"))
    parser.add_argument("--output", type=Path, default=RUNS / "extra_metrics")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    run_saved(args.output) if args.part == "saved" else run_heads(args.output, args.epochs)


if __name__ == "__main__":
    main()
