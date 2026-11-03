"""Auditable saliency router: frozen historical head versus a saliency-adapted head, on cached DINOv2-L tokens.

This is the v7 routing idea rebuilt on the fMoW study with its two known flaws removed:
  * checkpoint selection and router calibration use DISJOINT, location-grouped halves of adapt-val
    (v7 reused the same 2011 labels for both);
  * three seeds, with location-clustered bootstrap intervals (v7 had one seed).

For each image both models give a class distribution and a patch-weight distribution:
  historical  frozen linear head on mean-pooled tokens; patch weights = softmax of its own class evidence z
  updated     saliency-essential attention head (input or gated) from saliency_attention_study

Four named routing signals (as in v7):
  saliency_js        Jensen-Shannon divergence between the two patch-weight distributions
  confidence_gain    max softmax probability, updated minus historical
  entropy_reduction  predictive entropy, historical minus updated
  evidence_gain      selected-class logit from high-weight minus low-weight patches, updated minus historical

The updated prediction is used only when the two models disagree, saliency_js exceeds a stored floor and a
stored logistic probability exceeds a stored threshold. Every coefficient, mean, scale, floor and threshold is
written to policy JSON, and every test decision to a CSV row, so any single route can be recomputed by hand.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from fyp_drift.evaluation.fmow_dinov2l_study import DEFAULT_MANIFEST, cluster_bootstrap, load_rows, log, scores, study_splits, write_json
from fyp_drift.evaluation.saliency_adapters import SPLITS, batches
from fyp_drift.evaluation.saliency_attention_study import train

FEATURES = ("saliency_js", "confidence_gain", "entropy_reduction", "evidence_gain")
POLICIES = {"combined": FEATURES, "confidence_only": ("confidence_gain", "entropy_reduction"), "saliency_only": ("saliency_js", "evidence_gain")}
DELETE_FRACTION = 0.2


def selected_vs_complement(head, tokens, weights, classes):
    """Logit of the chosen class from high-weight patches minus the same logit from the low-weight complement."""
    complement = (1 - weights) / (1 - weights).sum(1, keepdim=True)
    high = head((weights[..., None] * tokens).sum(1))
    low = head((complement[..., None] * tokens).sum(1))
    return (high - low).gather(1, classes[:, None]).squeeze(1)


@torch.no_grad()
def paired_views(model, tokens, labels):
    """Historical and updated probabilities, patch weights and class evidence for every image, in order."""
    generator, views = torch.Generator().manual_seed(7), defaultdict(list)
    for x, _ in batches(tokens, labels, 256):
        z = model.saliency(x, generator)
        weights_old, weights_new = z.softmax(1), model.weights(x, z)
        probs_old, probs_new = model.frozen(x.mean(1)).softmax(1), model.classify(x, weights_new).softmax(1)
        views["probs_old"].append(probs_old.cpu())
        views["probs_new"].append(probs_new.cpu())
        views["weights_old"].append(weights_old.cpu())
        views["weights_new"].append(weights_new.cpu())
        views["evidence_old"].append(selected_vs_complement(model.frozen, x, weights_old, probs_old.argmax(1)).cpu())
        views["evidence_new"].append(selected_vs_complement(model.classifier, x, weights_new, probs_new.argmax(1)).cpu())
    return {k: torch.cat(v) for k, v in views.items()}


def routing_signals(views):
    entropy = lambda p: -(p * p.clamp_min(1e-12).log()).sum(1)
    kl = lambda a, b: (a * (a.clamp_min(1e-12) / b.clamp_min(1e-12)).log()).sum(1)
    middle = 0.5 * (views["weights_old"] + views["weights_new"])
    signals = {
        "saliency_js": 0.5 * kl(views["weights_old"], middle) + 0.5 * kl(views["weights_new"], middle),
        "confidence_gain": views["probs_new"].max(1).values - views["probs_old"].max(1).values,
        "entropy_reduction": entropy(views["probs_old"]) - entropy(views["probs_new"]),
        "evidence_gain": views["evidence_new"] - views["evidence_old"],
    }
    return {k: v.numpy() for k, v in signals.items()}


def fit_policy(name, signals, old, new, labels, classes):
    """Logistic update score fitted where the models disagree; floor and threshold searched on the same calibration half."""
    names, disagree = POLICIES[name], old != new
    X = np.column_stack([signals[n] for n in names])
    mean, scale = X[disagree].mean(0), X[disagree].std(0) + 1e-8
    target = (old != labels) & (new == labels)
    policy = {"name": name, "features": list(names), "mean": mean.tolist(), "scale": scale.tolist(),
              "calibration_images": int(len(labels)), "calibration_disagreements": int(disagree.sum()), "calibration_helpful": int(target[disagree].sum())}
    if len(set(target[disagree])) < 2:
        policy |= {"coefficients": [0.0] * len(names), "intercept": 0.0 if target[disagree].any() else -10.0, "floor": 0.0, "threshold": 0.5, "note": "one class only in calibration"}
        return policy
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(((X - mean) / scale)[disagree], target[disagree])
    policy |= {"coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}
    floors = [0.0] if name == "confidence_only" else [float(np.quantile(signals["saliency_js"][disagree], q)) for q in (0.0, 0.1, 0.25, 0.5)]
    best = None
    for floor in floors:
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
            routed = route(policy | {"floor": floor, "threshold": threshold}, signals, old, new)[0]
            final = np.where(routed, new, old)
            key = (scores(labels, final, classes)["macro_f1"], -int((routed & (old == labels) & (new != labels)).sum()))
            if best is None or key > best[0]:
                best = (key, floor, threshold)
    return policy | {"floor": best[1], "threshold": best[2], "calibration_macro_f1": best[0][0]}


def route(policy, signals, old, new):
    """Apply a stored policy: disagree AND saliency_js >= floor AND p(update) >= threshold."""
    z = (np.column_stack([signals[n] for n in policy["features"]]) - np.array(policy["mean"])) / np.array(policy["scale"])
    probability = 1 / (1 + np.exp(-(z @ np.array(policy["coefficients"]) + policy["intercept"])))
    eligible = (old != new) & (signals["saliency_js"] >= policy["floor"])
    return eligible & (probability >= policy["threshold"]), probability, z, eligible


@torch.no_grad()
def deletion_margin(model, tokens, labels, weights, historical, indices):
    """True-class probability drop after deleting the top 20% minus after deleting the bottom 20% of each model's own weights."""
    if len(indices) == 0:
        return np.array([])
    x = torch.from_numpy(np.asarray(tokens[np.sort(indices)])).cuda().float()
    order_index = np.argsort(np.argsort(indices))
    x, w, y = x[order_index], weights[indices].cuda(), torch.from_numpy(labels[indices]).cuda()
    k = round(DELETE_FRACTION * w.shape[1])

    def true_prob(keep):
        if historical:
            return model.frozen((keep[..., None] * x).sum(1) / keep.sum(1, keepdim=True)).softmax(1).gather(1, y[:, None]).squeeze(1)
        kept = w * keep
        return model.classify(x, kept / kept.sum(1, keepdim=True)).softmax(1).gather(1, y[:, None]).squeeze(1)

    full = true_prob(torch.ones_like(w))
    drop_top = full - true_prob(torch.ones_like(w).scatter_(1, w.argsort(1, descending=True)[:, :k], 0.0))
    drop_bottom = full - true_prob(torch.ones_like(w).scatter_(1, w.argsort(1)[:, :k], 0.0))
    return (drop_top - drop_bottom).cpu().numpy()


def outcome(old, new, final, label):
    if old != label and final == label:
        return "helpful_route"
    if old == label and final != label:
        return "harmful_route"
    if old != label and new == label:
        return "missed_correction"
    if old == label and new != label:
        return "harm_prevented"
    return "agree" if old == new else "both_wrong"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("/mnt/sdd/FYP/runs/today/main"))
    parser.add_argument("--tokens", type=Path, default=Path("/mnt/sdd/FYP/cache/adapter_tokens"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/sdd/FYP/runs/today/saliency_router"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--arms", nargs="+", default=["input", "gated"])
    parser.add_argument("--conditions", nargs="+", default=["clean", "cloud"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    train_args = SimpleNamespace(epochs=args.epochs, batch=128, lr=1e-3)
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    test_rows = splits["future_test"]
    labels_test = np.array([r["class_id"] for r in test_rows])
    groups_test = np.array([r["group"] for r in test_rows])
    val_groups = np.array([r["group"] for r in splits["adapt_val"]])
    select_idx, calibrate_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=7).split(val_groups, groups=val_groups))
    head_state = torch.load(args.run / "head.pt", weights_only=True)
    width = head_state["weight"].shape[1]
    for condition in args.conditions:
        tokens = {s: np.load(args.tokens / f"{condition}_{s}.npy", mmap_mode="r") for s in SPLITS}
        labels = {s: np.array([r["class_id"] for r in splits[s]]) for s in SPLITS}
        select = (np.asarray(tokens["adapt_val"][np.sort(select_idx)]), labels["adapt_val"][np.sort(select_idx)])
        calibrate = (np.asarray(tokens["adapt_val"][np.sort(calibrate_idx)]), labels["adapt_val"][np.sort(calibrate_idx)])
        for seed in args.seeds:
            for arm in args.arms:
                stem = f"{condition}_{arm}_s{seed}"
                if (args.output / f"{stem}.json").exists():
                    continue
                model, best = train(arm, seed, head_state, width, classes, {"adapt_train": (tokens["adapt_train"], labels["adapt_train"]), "adapt_val": select}, 0.0, train_args)
                cal_views = paired_views(model, *calibrate)
                cal_signals = routing_signals(cal_views)
                cal_old, cal_new = cal_views["probs_old"].argmax(1).numpy(), cal_views["probs_new"].argmax(1).numpy()
                test_views = paired_views(model, tokens["future_test"], labels_test)
                signals = routing_signals(test_views)
                old, new = test_views["probs_old"].argmax(1).numpy(), test_views["probs_new"].argmax(1).numpy()
                result = {"condition": condition, "arm": arm, "seed": seed, "best_epoch": best["epoch"],
                          "selection_images": int(len(select[1])), "calibration_images": int(len(calibrate[1])),
                          "historical": scores(labels_test, old, classes), "updated": scores(labels_test, new, classes), "policies": {}}
                for name in POLICIES:
                    policy = fit_policy(name, cal_signals, cal_old, cal_new, calibrate[1], classes)
                    write_json(args.output / f"{stem}_{name}_policy.json", policy)
                    routed, probability, z, eligible = route(policy, signals, old, new)
                    final = np.where(routed, new, old)
                    kinds = np.array([outcome(o, n, f, y) for o, n, f, y in zip(old, new, final, labels_test)])
                    routed_score = scores(labels_test, final, classes)
                    gain_new = result["updated"]["macro_f1"] - result["historical"]["macro_f1"]
                    entry = {"routed": routed_score, "coverage": float(routed.mean()),
                             "retained_gain": (routed_score["macro_f1"] - result["historical"]["macro_f1"]) / gain_new if abs(gain_new) > 1e-9 else None,
                             "counts": {k: int((kinds == k).sum()) for k in ("helpful_route", "harmful_route", "missed_correction", "harm_prevented")},
                             "routed_minus_updated_accuracy": cluster_bootstrap((final == labels_test).astype(float) - (new == labels_test).astype(float), groups_test, repetitions=1000),
                             "routed_minus_historical_accuracy": cluster_bootstrap((final == labels_test).astype(float) - (old == labels_test).astype(float), groups_test, repetitions=1000)}
                    if name == "combined":
                        for kind in ("helpful_route", "harmful_route"):
                            idx = np.flatnonzero(kinds == kind)
                            entry[f"deletion_margin_{kind}"] = {
                                "updated_map": float(np.mean(deletion_margin(model, tokens["future_test"], labels_test, test_views["weights_new"], False, idx))) if len(idx) else None,
                                "historical_map": float(np.mean(deletion_margin(model, tokens["future_test"], labels_test, test_views["weights_old"], True, idx))) if len(idx) else None,
                                "images": int(len(idx))}
                        with open(args.output / f"{stem}_route_log.csv", "w", newline="") as handle:
                            writer = csv.writer(handle)
                            writer.writerow(["image", "label", "historical_pred", "updated_pred", *FEATURES, *[f"z_{n}" for n in policy["features"]],
                                             "p_update", "disagree", "above_saliency_floor", "routed", "final_pred", "correct", "outcome"])
                            for i, r in enumerate(test_rows):
                                writer.writerow([r["local_path"], labels_test[i], old[i], new[i], *[f"{signals[n][i]:.6f}" for n in FEATURES],
                                                 *[f"{v:.4f}" for v in z[i]], f"{probability[i]:.4f}", int(old[i] != new[i]), int(eligible[i]),
                                                 int(routed[i]), final[i], int(final[i] == labels_test[i]), kinds[i]])
                    result["policies"][name] = entry
                write_json(args.output / f"{stem}.json", result)
                c = result["policies"]["combined"]
                log(f"{stem}: hist={result['historical']['macro_f1']:.4f} new={result['updated']['macro_f1']:.4f} routed={c['routed']['macro_f1']:.4f} "
                    f"cov={c['coverage']:.3f} counts={c['counts']}")
    summarize(args)


def summarize(args):
    runs = [json.loads(p.read_text()) for p in sorted(args.output.glob("*_s[0-9]*.json")) if "policy" not in p.name]
    table = {}
    for r in runs:
        for name, entry in r["policies"].items():
            key = f"{r['condition']}/{r['arm']}/{name}"
            t = table.setdefault(key, defaultdict(list))
            t["historical_f1"].append(r["historical"]["macro_f1"])
            t["updated_f1"].append(r["updated"]["macro_f1"])
            t["routed_f1"].append(entry["routed"]["macro_f1"])
            t["coverage"].append(entry["coverage"])
            if entry["retained_gain"] is not None:
                t["retained_gain"].append(entry["retained_gain"])
            for k, v in entry["counts"].items():
                t[k].append(v)
            for kind in ("helpful_route", "harmful_route"):
                margins = entry.get(f"deletion_margin_{kind}")
                if margins and margins["updated_map"] is not None:
                    t[f"updated_map_margin_{kind}"].append(margins["updated_map"])
                    t[f"historical_map_margin_{kind}"].append(margins["historical_map"])
    summary = {k: {m: float(np.mean(v)) for m, v in t.items()} | {"seeds": len(t["routed_f1"])} for k, t in table.items()}
    write_json(args.output / "summary.json", summary)
    log("summary", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
