"""Adapt, select, calibrate, and audit the twin-ViT saliency router."""
import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data import open_dataset, year_loader
from drift import (REALISTIC_BENCHMARK_SCENARIOS, SCENARIOS, drift_transform,
                   mixed_drift_transform, realistic_mixed_drift_transform,
                   scenario_components)
from metrics import binary_metrics, classification_metrics, topk_map_agreement
from model import TemporalViT
from run_store import RunStore
from siamese import ViTTwinRouter
from transparent_router import FEATURES, apply_policy, feature_tensor, fit_policy


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
def calibrate_router(router, loader, device):
    router.eval()
    features, beneficial, old_all, new_all, truth = [], [], [], [], []
    for images, labels, _ in tqdm(loader, desc="calibrate routing", unit="batch", leave=False):
        comparison = router.compare(images.to(device))
        old = comparison["old_logits"].argmax(1).cpu().numpy()
        new = comparison["new_logits"].argmax(1).cpu().numpy()
        actual = labels.numpy()
        features.extend(feature_tensor(comparison).cpu().numpy().tolist())
        beneficial.extend(((old != actual) & (new == actual)).astype(int).tolist())
        old_all.extend(old.tolist()); new_all.extend(new.tolist()); truth.extend(actual.tolist())
    return fit_policy(features, beneficial, np.asarray(old_all), np.asarray(new_all), np.asarray(truth))


@torch.no_grad()
def evaluate_stream(router, loader, policy, device, year_column):
    router.eval()
    decisions = []
    if device == "cuda":
        torch.cuda.synchronize()
        gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    wall_start = time.perf_counter()
    for images, labels, metadata in tqdm(loader, desc="future routing", unit="batch"):
        comparison = router.compare(images.to(device))
        routed, needs_update, use_new, route_probability, features = apply_policy(comparison, policy)
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
                "route_probability": round(float(route_probability[index]), 6),
                "route_threshold": policy["probability_threshold"],
                "old_evidence": round(float(comparison["old_evidence"][index]), 6),
                "new_evidence": round(float(comparison["new_evidence"][index]), 6),
                **{name: round(float(features[index, position]), 6)
                   for position, name in enumerate(FEATURES)},
                "needs_update": bool(needs_update[index]),
                "routed_to_new": bool(use_new[index]),
            })
    if device == "cuda":
        gpu_end.record()
        torch.cuda.synchronize()
        gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000
    else:
        gpu_seconds = None
    wall_seconds = time.perf_counter() - wall_start
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
    report["_timing"] = {"samples": len(decisions), "batches": len(loader),
                         "wall_seconds": round(wall_seconds, 4),
                         "gpu_seconds": round(gpu_seconds, 4) if gpu_seconds is not None else None,
                         "images_per_second": round(len(decisions) / wall_seconds, 2),
                         "milliseconds_per_image": round(1000 * wall_seconds / len(decisions), 4)}
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
    parser.add_argument("--evaluate-drift", action="store_true",
                        help="Opt in to controlled 2014 corruption evaluation")
    parser.add_argument("--drift-scenarios", nargs="+", choices=SCENARIOS,
                        default=list(REALISTIC_BENCHMARK_SCENARIOS),
                        help="Subset of drift conditions to evaluate; useful for controlled ablations")
    parser.add_argument("--adapt-year", type=int, default=2012)
    parser.add_argument("--calibration-year", type=int, default=2013)
    parser.add_argument("--test-years", type=int, nargs="+", default=(2014,))
    parser.add_argument("--long-horizon", action="store_true",
                        help="Use drift-aware 2010 adaptation, 2011 calibration, and frozen 2012-2014 tests")
    parser.add_argument("--adaptation-profile", choices=("standard", "enhanced", "realistic"),
                        default="standard",
                        help="Realistic uses acquisition, pipeline, viewpoint, and rare occlusion shifts")
    parser.add_argument("--disable-saliency-loss", action="store_true",
                        help="Ablation: retain the enhanced data/capacity but remove both saliency losses")
    parser.add_argument("--resume-run-dir", type=Path,
                        help="Resume the next epoch from models/resume.pt in an interrupted run")
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
    if args.long_horizon:
        args.adapt_year, args.calibration_year, args.test_years = 2010, 2011, (2012, 2013, 2014)
        if args.adaptation_profile == "realistic":
            adapt_transform = realistic_mixed_drift_transform((1, 2, 3), args.seed)
            calibration_transform = realistic_mixed_drift_transform(2, args.seed)
        else:
            adapt_transform = (mixed_drift_transform((1, 2, 3), args.seed, clean_repeats=1)
                               if args.adaptation_profile == "enhanced"
                               else mixed_drift_transform(1, args.seed))
            calibration_transform = mixed_drift_transform(2, args.seed)
        adapt_split = calibration_split = "train_source"
        test_split = "test_source"
    else:
        adapt_transform = calibration_transform = None
        adapt_split, calibration_split, test_split = "val", "val", "test"
    adapt, _ = year_loader(dataset, adapt_split, (args.adapt_year,), args.batch_size,
                           shuffle=True, limit=args.limit_adapt, seed=args.seed,
                           image_transform=adapt_transform or drift_transform("clean", 1, args.seed))
    calibrate, _ = year_loader(dataset, calibration_split, (args.calibration_year,), args.batch_size,
                               limit=args.limit_calibrate,
                               image_transform=calibration_transform or drift_transform("clean", 1, args.seed))
    future, year_column = year_loader(dataset, test_split, args.test_years, args.batch_size,
                                      limit=args.limit_test,
                                      image_transform=drift_transform("clean", 1, args.seed))
    if not len(adapt.dataset) or not len(calibrate.dataset) or not len(future.dataset):
        raise ValueError("One year split is empty; check the CLEAR-100 manifest")
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
    strong_profile = args.adaptation_profile in ("enhanced", "realistic")
    trainable_blocks = 3 if strong_profile else 1
    for block in new.backbone.encoder.layer[-trainable_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    router = ViTTwinRouter(old, new)
    backbone_parameters = [p for p in new.backbone.parameters() if p.requires_grad]
    head_parameters = list(new.saliency_head.parameters()) + list(new.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": 1e-5},
        {"params": head_parameters, "lr": 5e-5 if strong_profile else 1e-5},
    ], weight_decay=0.01)
    start_epoch, best_score = 1, -1.0
    if args.resume_run_dir:
        store = RunStore.from_existing(args.resume_run_dir)
        resume = torch.load(store.path / "models" / "resume.pt", map_location=device, weights_only=False)
        new.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch, best_score = int(resume["epoch"]) + 1, float(resume["best_score"])
        store.log({"event": "resume", "from_epoch": resume["epoch"], "next_epoch": start_epoch})
    else:
        store = RunStore(args.output, "siamese", args.run_id)
        store.write_json("config.json", vars(args) | {"dataset": "CLEAR-100", "num_classes": classes,
                                                      "device": device, "torch_version": torch.__version__,
                                                      "cuda_build": torch.version.cuda})
    for epoch in range(start_epoch, args.epochs + 1):
        new.train()
        total = 0.0
        if device == "cuda":
            torch.cuda.synchronize()
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        wall_start = time.perf_counter()
        progress = tqdm(adapt, desc=f"siamese epoch {epoch}/{args.epochs}", unit="batch")
        for step, (images, labels, _) in enumerate(progress, 1):
            labels = labels.to(device)
            comparison = router.compare(images.to(device))
            logits = comparison["new_logits"]
            gap = comparison["saliency_gap"]
            old_correct = comparison["old_logits"].argmax(1).eq(labels)
            margin = 0.08 if strong_profile else 0.05
            pair_loss = torch.where(old_correct, gap, F.relu(margin - gap)).mean()
            true_evidence = (logits.softmax(1) - comparison["new_inverse"].softmax(1)).gather(1, labels[:, None])
            saliency_weight = (0.0 if args.disable_saliency_loss else
                               (0.25 if strong_profile else 0.1))
            loss = (F.cross_entropy(logits, labels,
                                    label_smoothing=0.05 if strong_profile else 0.0)
                    + saliency_weight * pair_loss
                    + saliency_weight * F.relu(0.03 - true_evidence).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step % args.log_every == 0 or step == len(adapt):
                store.log({"event": "adapt_batch", "epoch": epoch, "batch": step,
                           "batches": len(adapt), "loss": round(loss.item(), 6)})
        if device == "cuda":
            gpu_end.record()
            torch.cuda.synchronize()
            gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000
        else:
            gpu_seconds = None
        wall_seconds = time.perf_counter() - wall_start
        validation = model_metrics(new, calibrate, device)
        score = validation["macro_f1"]
        metric_name = f"calibration_{args.calibration_year}_macro_f1"
        row = {"epoch": epoch, "adapt_loss": round(total / len(adapt.dataset), 4),
               metric_name: score,
               f"calibration_{args.calibration_year}_macro_iou": validation["macro_iou"]}
        row["timing"] = {"adapt_wall_seconds": round(wall_seconds, 4),
                         "adapt_gpu_seconds": round(gpu_seconds, 4) if gpu_seconds is not None else None,
                         "adapt_images_per_second": round(len(adapt.dataset) / wall_seconds, 2)}
        store.log(row)
        store.save_model(new, "last", epoch, metric_name, score)
        if score > best_score:
            best_score = score
            store.save_model(new, "best", epoch, metric_name, score)
        store.save_training_checkpoint(new, optimizer, epoch, best_score)
        print(json.dumps(row), flush=True)
    new.load_state_dict(torch.load(store.path / "models" / "best.pt", map_location=device, weights_only=True))
    policy = calibrate_router(router, calibrate, device)
    store.write_json("metrics/router_policy.json", policy)
    result, decisions = evaluate_stream(router, future, policy, device, year_column)
    store.write_json("metrics/future_test.json", result)
    with (store.path / "logs" / "routing_decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    store.log({"event": "final", "selected_checkpoint": "models/best.pt",
               "selection_metric": f"calibration_{args.calibration_year}_macro_f1", "best_score": best_score,
               "future_routed_macro_f1": result["all"]["routed"]["macro_f1"]})
    if args.evaluate_drift:
        drift_results = {}
        drift_decisions = []
        for scenario in args.drift_scenarios:
            drift_results[scenario] = {}
            for severity, year in enumerate(args.test_years, 1):
                progress_event = {"event": "drift_eval_start", "scenario": scenario,
                                  "year": year, "severity": 0 if scenario == "clean" else severity}
                store.log(progress_event)
                print(json.dumps(progress_event), flush=True)
                subset = dataset.get_subset(test_split, transform=drift_transform(scenario, severity, args.seed))
                indices = torch.where(subset.metadata_array[:, year_column] == year - 2002)[0].tolist()
                if args.limit_test:
                    indices = indices[:args.limit_test]
                loader = DataLoader(Subset(subset, indices), batch_size=args.batch_size)
                scenario_report, scenario_decisions = evaluate_stream(router, loader, policy, device, year_column)
                for dataset_index, row in zip(indices, scenario_decisions):
                    drift_decisions.append({"scenario": scenario,
                                            "severity": 0 if scenario == "clean" else severity,
                                            "dataset_index": dataset_index,
                                            "drift_components": "+".join(scenario_components(
                                                scenario, dataset_index, severity, args.seed)),
                                            **row})
                drift_results[scenario][str(year)] = {
                    "severity": 0 if scenario == "clean" else severity,
                    "results": scenario_report}
                completed_event = {
                    "event": "drift_eval_complete", "scenario": scenario, "year": year,
                    "severity": 0 if scenario == "clean" else severity,
                    "old_macro_f1": scenario_report["all"]["old"]["macro_f1"],
                    "new_macro_f1": scenario_report["all"]["new"]["macro_f1"],
                    "routed_macro_f1": scenario_report["all"]["routed"]["macro_f1"],
                    "helpful_routes": scenario_report["all"]["helpful_routes"],
                    "harmful_routes": scenario_report["all"]["harmful_routes"],
                }
                store.log(completed_event)
                print(json.dumps(completed_event), flush=True)
                store.write_json("metrics/future_drift.partial.json", {
                    "compound_shift_held_out": True, "test_years": args.test_years,
                    "scenarios": drift_results})
        store.write_json("metrics/future_drift.json", {
            "compound_shift_held_out": True, "test_years": args.test_years,
            "scenarios": drift_results})
        with (store.path / "logs" / "drift_routing_decisions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(drift_decisions[0]))
            writer.writeheader()
            writer.writerows(drift_decisions)
    print(json.dumps({"run": str(store.path), "future": {
        key: result["all"][key]["macro_f1"] for key in ("old", "new", "routed")}}, indent=2))


if __name__ == "__main__":
    main()
