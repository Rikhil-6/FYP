"""Adapt, select, calibrate, and audit the twin-ViT saliency router."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from data import open_dataset, year_loader
from metrics import binary_metrics, classification_metrics, topk_map_agreement
from model import TemporalViT
from run_store import RunStore
from siamese import ViTTwinRouter, fit_thresholds


@torch.no_grad()
def model_metrics(model, loader, device):
    model.eval()
    truth, prediction = [], []
    for images, labels, _ in tqdm(loader, desc="select newer ViT", unit="batch", leave=False):
        logits = model(images.to(device))[0]
        truth.extend(labels.tolist())
        prediction.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(truth, prediction)


@torch.no_grad()
def calibrate_thresholds(router, loader, device):
    router.eval()
    class_scores = defaultdict(list)
    for images, labels, _ in tqdm(loader, desc="calibrate routing", unit="batch", leave=False):
        comparison = router.compare(images.to(device))
        old_prediction = comparison["old_logits"].argmax(1).cpu()
        for index, class_id in enumerate(old_prediction.tolist()):
            if class_id == int(labels[index]):
                class_scores[class_id].append(float(comparison["saliency_gap"][index]))
    return fit_thresholds(class_scores)


@torch.no_grad()
def evaluate_stream(router, loader, thresholds, device, year_column):
    router.eval()
    decisions = []
    for images, labels, metadata in tqdm(loader, desc="future routing", unit="batch"):
        comparison = router.compare(images.to(device))
        routed, needs_update, use_new = router.route(comparison, thresholds)
        old = comparison["old_logits"].argmax(1)
        new = comparison["new_logits"].argmax(1)
        for index, year_id in enumerate(metadata[:, year_column].tolist()):
            old_class = int(old[index])
            overlap = topk_map_agreement(comparison["old_map"][index].cpu().numpy(),
                                         comparison["new_map"][index].cpu().numpy())
            decisions.append({
                "year": 2002 + year_id, "true_class": int(labels[index]),
                "old_class": old_class, "new_class": int(new[index]),
                "routed_class": int(routed[index]),
                "saliency_js": round(float(comparison["saliency_gap"][index]), 6),
                "saliency_top20_iou": overlap["iou"],
                "saliency_top20_f1": overlap["f1"],
                "class_threshold": round(thresholds.get(old_class, thresholds["global"]), 6),
                "old_evidence": round(float(comparison["old_evidence"][index]), 6),
                "new_evidence": round(float(comparison["new_evidence"][index]), 6),
                "needs_update": bool(needs_update[index]),
                "routed_to_new": bool(use_new[index]),
            })
    grouped = defaultdict(list)
    for row in decisions:
        grouped["all"].append(row)
        grouped[str(row["year"])].append(row)
    report = {}
    for year, rows in grouped.items():
        truth = [row["true_class"] for row in rows]
        beneficial = [row["old_class"] != row["true_class"] and
                      row["new_class"] == row["true_class"] for row in rows]
        flagged = [row["needs_update"] for row in rows]
        report[year] = {
            "old": classification_metrics(truth, [row["old_class"] for row in rows]),
            "new": classification_metrics(truth, [row["new_class"] for row in rows]),
            "routed": classification_metrics(truth, [row["routed_class"] for row in rows]),
            "update_trigger_vs_beneficial_switch": binary_metrics(beneficial, flagged),
            "update_flags": sum(flagged),
            "routed_to_new": sum(row["routed_to_new"] for row in rows),
            "helpful_routes": sum(row["routed_to_new"] and row["old_class"] != row["true_class"]
                                  and row["new_class"] == row["true_class"] for row in rows),
            "harmful_routes": sum(row["routed_to_new"] and row["old_class"] == row["true_class"]
                                  and row["new_class"] != row["true_class"] for row in rows),
            "mean_saliency_js": round(float(np.mean([row["saliency_js"] for row in rows])), 6),
            "mean_saliency_top20_iou": round(float(np.mean([row["saliency_top20_iou"] for row in rows])), 4),
            "mean_saliency_top20_f1": round(float(np.mean([row["saliency_top20_f1"] for row in rows])), 4),
        }
    return report, decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-adapt", type=int)
    parser.add_argument("--limit-calibrate", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = open_dataset(args.data_root)
    adapt, _ = year_loader(dataset, "val", (2012,), args.batch_size,
                           shuffle=True, limit=args.limit_adapt, seed=args.seed)
    calibrate, _ = year_loader(dataset, "val", (2013,), args.batch_size,
                               limit=args.limit_calibrate)
    future, year_column = year_loader(dataset, "test", (2014,), args.batch_size,
                                      limit=args.limit_test)
    if not len(adapt.dataset) or not len(calibrate.dataset) or not len(future.dataset):
        raise ValueError("One year split is empty; check FMoW metadata")
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint["classifier.weight"].shape[0]
    old, new = TemporalViT("xai", classes).to(device), TemporalViT("xai", classes).to(device)
    old.load_state_dict(checkpoint)
    new.load_state_dict(checkpoint)
    old.eval()
    for parameter in old.parameters():
        parameter.requires_grad = False
    for parameter in new.backbone.parameters():
        parameter.requires_grad = False
    for parameter in new.backbone.encoder.layer[-1].parameters():
        parameter.requires_grad = True
    router = ViTTwinRouter(old, new)
    optimizer = torch.optim.AdamW((p for p in new.parameters() if p.requires_grad), lr=1e-5)
    store = RunStore(args.output, "siamese", args.run_id)
    store.write_json("config.json", vars(args) | {"dataset": "CLEAR-100", "num_classes": classes,
                                                  "device": device, "torch_version": torch.__version__,
                                                  "cuda_build": torch.version.cuda})
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        new.train()
        total = 0.0
        progress = tqdm(adapt, desc=f"siamese epoch {epoch}/{args.epochs}", unit="batch")
        for step, (images, labels, _) in enumerate(progress, 1):
            labels = labels.to(device)
            comparison = router.compare(images.to(device))
            logits = comparison["new_logits"]
            gap = comparison["saliency_gap"]
            old_correct = comparison["old_logits"].argmax(1).eq(labels)
            pair_loss = torch.where(old_correct, gap, F.relu(0.05 - gap)).mean()
            true_evidence = (logits.softmax(1) - comparison["new_inverse"].softmax(1)).gather(1, labels[:, None])
            loss = F.cross_entropy(logits, labels) + 0.1 * pair_loss + 0.1 * F.relu(0.02 - true_evidence).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step % args.log_every == 0 or step == len(adapt):
                store.log({"event": "adapt_batch", "epoch": epoch, "batch": step,
                           "batches": len(adapt), "loss": round(loss.item(), 6)})
        validation = model_metrics(new, calibrate, device)
        score = validation["macro_f1"]
        row = {"epoch": epoch, "adapt_loss": round(total / len(adapt.dataset), 4),
               "calibration_2013_macro_f1": score,
               "calibration_2013_macro_iou": validation["macro_iou"]}
        store.log(row)
        store.save_model(new, "last", epoch, "calibration_2013_macro_f1", score)
        if score > best_score:
            best_score = score
            store.save_model(new, "best", epoch, "calibration_2013_macro_f1", score)
        print(json.dumps(row), flush=True)
    new.load_state_dict(torch.load(store.path / "models" / "best.pt", map_location=device, weights_only=True))
    thresholds = calibrate_thresholds(router, calibrate, device)
    store.write_json("metrics/thresholds.json", thresholds)
    result, decisions = evaluate_stream(router, future, thresholds, device, year_column)
    store.write_json("metrics/future_test.json", result)
    with (store.path / "logs" / "routing_decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    store.log({"event": "final", "selected_checkpoint": "models/best.pt",
               "selection_metric": "calibration_2013_macro_f1", "best_score": best_score,
               "future_routed_macro_f1": result["all"]["routed"]["macro_f1"]})
    print(json.dumps({"run": str(store.path), "future": {
        key: result["all"][key]["macro_f1"] for key in ("old", "new", "routed")}}, indent=2))


if __name__ == "__main__":
    main()
