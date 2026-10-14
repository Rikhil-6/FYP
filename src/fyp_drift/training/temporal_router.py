"""Adapt a ViT snapshot and calibrate the transparent temporal router."""

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from fyp_drift.artifacts.run_store import RunStore
from fyp_drift.data.clear100 import (
    open_dataset,
    stratified_year_partition_loaders,
    year_loader,
)
from fyp_drift.data.corruptions import (
    REALISTIC_BENCHMARK_SCENARIOS,
    SCENARIOS,
    drift_transform,
    mixed_drift_transform,
    realistic_mixed_drift_transform,
    scenario_components,
)
from fyp_drift.evaluation.metrics import (
    binary_metrics,
    classification_metrics,
    topk_map_agreement,
)
from fyp_drift.models.routing import (
    FEATURES,
    FEATURE_SETS,
    apply_policy,
    feature_tensor,
    fit_policy,
)
from fyp_drift.models.siamese import ViTTwinRouter
from fyp_drift.models.vision_transformer import TemporalViT


@dataclass(frozen=True)
class TemporalLoaders:
    """Data loaders and metadata needed by the temporal experiment."""

    adaptation: DataLoader
    selection: DataLoader
    calibration: DataLoader
    future: DataLoader
    year_column: int
    test_split: str
    calibration_protocol: str


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Adapt an XAI ViT and calibrate its temporal routing policy."
    )
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
    parser.add_argument(
        "--evaluate-drift",
        action="store_true",
        help="Evaluate the frozen router under controlled future corruptions.",
    )
    parser.add_argument(
        "--drift-scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(REALISTIC_BENCHMARK_SCENARIOS),
        help="Drift conditions to evaluate after the clean future stream.",
    )
    parser.add_argument("--adapt-year", type=int, default=2012)
    parser.add_argument("--calibration-year", type=int, default=2013)
    parser.add_argument(
        "--selection-fraction",
        type=float,
        default=0.5,
        help="Class-stratified calibration-year share used for checkpoint selection.",
    )
    parser.add_argument(
        "--reuse-calibration-for-selection",
        action="store_true",
        help="Reproduce archived runs that selected and calibrated on the same data.",
    )
    parser.add_argument("--test-years", type=int, nargs="+", default=(2014,))
    parser.add_argument(
        "--long-horizon",
        action="store_true",
        help="Adapt in 2010, calibrate in 2011, and test the frozen router in 2012-2014.",
    )
    parser.add_argument(
        "--adaptation-profile",
        choices=("standard", "enhanced", "realistic"),
        default="standard",
        help="Choose the adaptation curriculum and trainable backbone depth.",
    )
    parser.add_argument(
        "--disable-saliency-loss",
        action="store_true",
        help="Ablate the pairwise and counterfactual saliency losses.",
    )
    parser.add_argument(
        "--router-features",
        choices=tuple(FEATURE_SETS),
        default="combined",
        help="Fit the route policy with prediction, explanation, or combined evidence.",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Continue from models/resume.pt inside an interrupted run.",
    )
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    if not 0 < args.selection_fraction < 1:
        parser.error("--selection-fraction must be strictly between 0 and 1")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_timeline(args):
    """Return split names and transforms for the requested protocol."""
    if not args.long_horizon:
        clean = drift_transform("clean", 1, args.seed)
        return "val", "val", "test", clean, clean

    args.adapt_year = 2010
    args.calibration_year = 2011
    args.test_years = (2012, 2013, 2014)
    if args.adaptation_profile == "realistic":
        adaptation_transform = realistic_mixed_drift_transform((1, 2, 3), args.seed)
        calibration_transform = realistic_mixed_drift_transform(2, args.seed)
    elif args.adaptation_profile == "enhanced":
        adaptation_transform = mixed_drift_transform(
            (1, 2, 3), args.seed, clean_repeats=1
        )
        calibration_transform = mixed_drift_transform(2, args.seed)
    else:
        adaptation_transform = mixed_drift_transform(1, args.seed)
        calibration_transform = mixed_drift_transform(2, args.seed)
    return (
        "train_source",
        "train_source",
        "test_source",
        adaptation_transform,
        calibration_transform,
    )


def build_temporal_loaders(dataset, args):
    """Build disjoint selection, calibration, and future data streams."""
    (
        adaptation_split,
        calibration_split,
        test_split,
        adaptation_transform,
        calibration_transform,
    ) = _resolve_timeline(args)

    adaptation, _ = year_loader(
        dataset,
        adaptation_split,
        (args.adapt_year,),
        args.batch_size,
        shuffle=True,
        limit=args.limit_adapt,
        seed=args.seed,
        image_transform=adaptation_transform,
    )
    if args.reuse_calibration_for_selection:
        calibration, _ = year_loader(
            dataset,
            calibration_split,
            (args.calibration_year,),
            args.batch_size,
            limit=args.limit_calibrate,
            image_transform=calibration_transform,
        )
        selection = calibration
        calibration_protocol = "shared_legacy"
    else:
        selection, calibration, _ = stratified_year_partition_loaders(
            dataset,
            calibration_split,
            args.calibration_year,
            args.batch_size,
            fraction=args.selection_fraction,
            limit=args.limit_calibrate,
            seed=args.seed,
            image_transform=calibration_transform,
        )
        calibration_protocol = "disjoint_class_stratified"

    future, year_column = year_loader(
        dataset,
        test_split,
        args.test_years,
        args.batch_size,
        limit=args.limit_test,
        image_transform=drift_transform("clean", 1, args.seed),
    )
    loaders = TemporalLoaders(
        adaptation=adaptation,
        selection=selection,
        calibration=calibration,
        future=future,
        year_column=year_column,
        test_split=test_split,
        calibration_protocol=calibration_protocol,
    )
    if any(
        len(loader.dataset) == 0
        for loader in (
            loaders.adaptation,
            loaders.selection,
            loaders.calibration,
            loaders.future,
        )
    ):
        raise ValueError("One temporal split is empty; check the CLEAR-100 manifest")
    return loaders


def validate_resume_protocol(run_dir, requested_protocol):
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_protocol = saved_config.get("calibration_protocol", "shared_legacy")
    if saved_protocol != requested_protocol:
        raise ValueError(
            f"Resume protocol mismatch: run uses {saved_protocol}, "
            f"requested {requested_protocol}."
        )


def build_router(base_checkpoint, device, adaptation_profile):
    """Load frozen historical and trainable updated copies of the XAI ViT."""
    state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    num_classes = state["classifier.weight"].shape[0]
    historical_model = TemporalViT("xai", num_classes).to(device)
    updated_model = TemporalViT("xai", num_classes).to(device)
    historical_model.load_state_dict(state)
    updated_model.load_state_dict(state)

    historical_model.eval()
    for parameter in historical_model.parameters():
        parameter.requires_grad = False
    for parameter in updated_model.backbone.parameters():
        parameter.requires_grad = False

    uses_strong_profile = adaptation_profile in ("enhanced", "realistic")
    trainable_blocks = 3 if uses_strong_profile else 1
    for block in updated_model.backbone.encoder.layer[-trainable_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    router = ViTTwinRouter(historical_model, updated_model)
    backbone_parameters = [
        parameter
        for parameter in updated_model.backbone.parameters()
        if parameter.requires_grad
    ]
    head_parameters = list(updated_model.saliency_head.parameters()) + list(
        updated_model.classifier.parameters()
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 1e-5},
            {
                "params": head_parameters,
                "lr": 5e-5 if uses_strong_profile else 1e-5,
            },
        ],
        weight_decay=0.01,
    )
    return router, optimizer, num_classes, uses_strong_profile


@torch.no_grad()
def model_metrics(model, loader, device):
    model.eval()
    truth = []
    predictions = []
    for images, labels, _ in tqdm(
        loader, desc="select updated ViT", unit="batch", leave=False
    ):
        logits = model(images.to(device))[0]
        truth.extend(labels.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(truth, predictions)


@torch.no_grad()
def calibrate_router(router, loader, device, feature_set="combined"):
    """Fit the transparent policy to beneficial model switches."""
    router.eval()
    feature_rows = []
    beneficial_switches = []
    historical_predictions = []
    updated_predictions = []
    truth = []
    for images, labels, _ in tqdm(
        loader, desc="calibrate routing", unit="batch", leave=False
    ):
        comparison = router.compare(images.to(device))
        historical = comparison["old_logits"].argmax(1).cpu().numpy()
        updated = comparison["new_logits"].argmax(1).cpu().numpy()
        actual = labels.numpy()
        feature_rows.extend(feature_tensor(comparison).cpu().numpy().tolist())
        beneficial_switches.extend(
            ((historical != actual) & (updated == actual)).astype(int).tolist()
        )
        historical_predictions.extend(historical.tolist())
        updated_predictions.extend(updated.tolist())
        truth.extend(actual.tolist())
    return fit_policy(
        feature_rows,
        beneficial_switches,
        np.asarray(historical_predictions),
        np.asarray(updated_predictions),
        np.asarray(truth),
        feature_names=FEATURE_SETS[feature_set],
    )


def _summarize_decisions(rows):
    truth = [row["true_class"] for row in rows]
    beneficial = [
        row["old_class"] != row["true_class"] and row["new_class"] == row["true_class"]
        for row in rows
    ]
    flagged = [row["needs_update"] for row in rows]
    return {
        "old": classification_metrics(truth, [row["old_class"] for row in rows]),
        "new": classification_metrics(truth, [row["new_class"] for row in rows]),
        "routed": classification_metrics(truth, [row["routed_class"] for row in rows]),
        "update_trigger_vs_beneficial_switch": binary_metrics(beneficial, flagged),
        "update_flags": sum(flagged),
        "routed_to_new": sum(row["routed_to_new"] for row in rows),
        "helpful_routes": sum(
            row["routed_to_new"]
            and row["old_class"] != row["true_class"]
            and row["new_class"] == row["true_class"]
            for row in rows
        ),
        "harmful_routes": sum(
            row["routed_to_new"]
            and row["old_class"] == row["true_class"]
            and row["new_class"] != row["true_class"]
            for row in rows
        ),
        "mean_saliency_js": round(
            float(np.mean([row["saliency_js"] for row in rows])), 6
        ),
        "mean_saliency_top20_iou": round(
            float(np.mean([row["saliency_top20_iou"] for row in rows])), 4
        ),
        "mean_saliency_top20_f1": round(
            float(np.mean([row["saliency_top20_f1"] for row in rows])), 4
        ),
    }


@torch.no_grad()
def evaluate_stream(router, loader, policy, device, year_column):
    """Evaluate each routing decision and retain its interpretable inputs."""
    router.eval()
    decisions = []
    if device == "cuda":
        torch.cuda.synchronize()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    wall_start = time.perf_counter()

    for images, labels, metadata in tqdm(loader, desc="future routing", unit="batch"):
        comparison = router.compare(images.to(device))
        routed, needs_update, use_updated, route_probability, features = apply_policy(
            comparison, policy
        )
        historical = comparison["old_logits"].argmax(1)
        updated = comparison["new_logits"].argmax(1)
        for index, year_id in enumerate(metadata[:, year_column].tolist()):
            overlap = topk_map_agreement(
                comparison["old_map"][index].cpu().numpy(),
                comparison["new_map"][index].cpu().numpy(),
            )
            decisions.append(
                {
                    "year": 2002 + year_id,
                    "true_class": int(labels[index]),
                    "old_class": int(historical[index]),
                    "new_class": int(updated[index]),
                    "routed_class": int(routed[index]),
                    "saliency_js": round(float(comparison["saliency_gap"][index]), 6),
                    "saliency_top20_iou": overlap["iou"],
                    "saliency_top20_f1": overlap["f1"],
                    "route_probability": round(float(route_probability[index]), 6),
                    "route_threshold": policy["probability_threshold"],
                    "old_evidence": round(float(comparison["old_evidence"][index]), 6),
                    "new_evidence": round(float(comparison["new_evidence"][index]), 6),
                    **{
                        name: round(float(features[index, position]), 6)
                        for position, name in enumerate(FEATURES)
                    },
                    "needs_update": bool(needs_update[index]),
                    "routed_to_new": bool(use_updated[index]),
                }
            )

    if device == "cuda":
        gpu_end.record()
        torch.cuda.synchronize()
        gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000
    else:
        gpu_seconds = None
    wall_seconds = time.perf_counter() - wall_start
    grouped = defaultdict(list)
    for decision in decisions:
        grouped["all"].append(decision)
        grouped[str(decision["year"])].append(decision)
    report = {name: _summarize_decisions(rows) for name, rows in grouped.items()}
    report["_timing"] = {
        "samples": len(decisions),
        "batches": len(loader),
        "wall_seconds": round(wall_seconds, 4),
        "gpu_seconds": round(gpu_seconds, 4) if gpu_seconds is not None else None,
        "images_per_second": round(len(decisions) / wall_seconds, 2),
        "milliseconds_per_image": round(1000 * wall_seconds / len(decisions), 4),
    }
    return report, decisions


def create_run_store(args, loaders, num_classes, device):
    if args.resume_run_dir:
        validate_resume_protocol(args.resume_run_dir, loaders.calibration_protocol)
        return RunStore.from_existing(args.resume_run_dir)

    store = RunStore(args.output, "siamese", args.run_id)
    store.write_json(
        "config.json",
        vars(args)
        | {
            "dataset": "CLEAR-100",
            "num_classes": num_classes,
            "device": device,
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "calibration_protocol": loaders.calibration_protocol,
            "selection_samples": len(loaders.selection.dataset),
            "router_calibration_samples": len(loaders.calibration.dataset),
        },
    )
    return store


def restore_training_state(args, store, updated_model, optimizer, device):
    if not args.resume_run_dir:
        return 1, -1.0
    resume = torch.load(
        store.path / "models" / "resume.pt",
        map_location=device,
        weights_only=False,
    )
    updated_model.load_state_dict(resume["model"])
    optimizer.load_state_dict(resume["optimizer"])
    start_epoch = int(resume["epoch"]) + 1
    best_score = float(resume["best_score"])
    store.log(
        {"event": "resume", "from_epoch": resume["epoch"], "next_epoch": start_epoch}
    )
    return start_epoch, best_score


def adapt_updated_model(
    router,
    optimizer,
    loaders,
    store,
    args,
    device,
    uses_strong_profile,
    start_epoch,
    best_score,
):
    """Train the updated tower and select its best temporal checkpoint."""
    updated_model = router.new
    saliency_weight = (
        0.0 if args.disable_saliency_loss else (0.25 if uses_strong_profile else 0.1)
    )
    saliency_margin = 0.08 if uses_strong_profile else 0.05
    label_smoothing = 0.05 if uses_strong_profile else 0.0
    metric_name = f"selection_{args.calibration_year}_macro_f1"

    for epoch in range(start_epoch, args.epochs + 1):
        updated_model.train()
        loss_sum = 0.0
        if device == "cuda":
            torch.cuda.synchronize()
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        wall_start = time.perf_counter()
        progress = tqdm(
            loaders.adaptation,
            desc=f"siamese epoch {epoch}/{args.epochs}",
            unit="batch",
        )
        for step, (images, labels, _) in enumerate(progress, 1):
            labels = labels.to(device)
            comparison = router.compare(images.to(device))
            updated_logits = comparison["new_logits"]
            historical_correct = comparison["old_logits"].argmax(1).eq(labels)
            pairwise_loss = torch.where(
                historical_correct,
                comparison["saliency_gap"],
                F.relu(saliency_margin - comparison["saliency_gap"]),
            ).mean()
            true_evidence = (
                updated_logits.softmax(1) - comparison["new_inverse"].softmax(1)
            ).gather(1, labels[:, None])
            loss = (
                F.cross_entropy(updated_logits, labels, label_smoothing=label_smoothing)
                + saliency_weight * pairwise_loss
                + saliency_weight * F.relu(0.03 - true_evidence).mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step % args.log_every == 0 or step == len(loaders.adaptation):
                store.log(
                    {
                        "event": "adapt_batch",
                        "epoch": epoch,
                        "batch": step,
                        "batches": len(loaders.adaptation),
                        "loss": round(loss.item(), 6),
                    }
                )

        if device == "cuda":
            gpu_end.record()
            torch.cuda.synchronize()
            gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000
        else:
            gpu_seconds = None
        wall_seconds = time.perf_counter() - wall_start
        validation = model_metrics(updated_model, loaders.selection, device)
        score = validation["macro_f1"]
        row = {
            "epoch": epoch,
            "adapt_loss": round(loss_sum / len(loaders.adaptation.dataset), 4),
            metric_name: score,
            f"selection_{args.calibration_year}_macro_iou": validation["macro_iou"],
            "timing": {
                "adapt_wall_seconds": round(wall_seconds, 4),
                "adapt_gpu_seconds": round(gpu_seconds, 4)
                if gpu_seconds is not None
                else None,
                "adapt_images_per_second": round(
                    len(loaders.adaptation.dataset) / wall_seconds, 2
                ),
            },
        }
        store.log(row)
        store.save_model(updated_model, "last", epoch, metric_name, score)
        if score > best_score:
            best_score = score
            store.save_model(updated_model, "best", epoch, metric_name, score)
        store.save_training_checkpoint(updated_model, optimizer, epoch, best_score)
        print(json.dumps(row), flush=True)
    return best_score


def write_decisions(path, decisions):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)


def evaluate_drift_scenarios(router, policy, dataset, loaders, store, args, device):
    scenario_results = {}
    all_decisions = []
    for scenario in args.drift_scenarios:
        scenario_results[scenario] = {}
        for severity, year in enumerate(args.test_years, 1):
            reported_severity = 0 if scenario == "clean" else severity
            started = {
                "event": "drift_eval_start",
                "scenario": scenario,
                "year": year,
                "severity": reported_severity,
            }
            store.log(started)
            print(json.dumps(started), flush=True)
            subset = dataset.get_subset(
                loaders.test_split,
                transform=drift_transform(scenario, severity, args.seed),
            )
            indices = torch.where(
                subset.metadata_array[:, loaders.year_column] == year - 2002
            )[0].tolist()
            if args.limit_test:
                indices = indices[: args.limit_test]
            stream = DataLoader(Subset(subset, indices), batch_size=args.batch_size)
            report, decisions = evaluate_stream(
                router, stream, policy, device, loaders.year_column
            )
            for dataset_index, decision in zip(indices, decisions):
                all_decisions.append(
                    {
                        "scenario": scenario,
                        "severity": reported_severity,
                        "dataset_index": dataset_index,
                        "drift_components": "+".join(
                            scenario_components(
                                scenario, dataset_index, severity, args.seed
                            )
                        ),
                        **decision,
                    }
                )
            scenario_results[scenario][str(year)] = {
                "severity": reported_severity,
                "results": report,
            }
            completed = {
                "event": "drift_eval_complete",
                "scenario": scenario,
                "year": year,
                "severity": reported_severity,
                "old_macro_f1": report["all"]["old"]["macro_f1"],
                "new_macro_f1": report["all"]["new"]["macro_f1"],
                "routed_macro_f1": report["all"]["routed"]["macro_f1"],
                "helpful_routes": report["all"]["helpful_routes"],
                "harmful_routes": report["all"]["harmful_routes"],
            }
            store.log(completed)
            print(json.dumps(completed), flush=True)
            store.write_json(
                "metrics/future_drift.partial.json",
                {
                    "compound_shift_held_out": True,
                    "test_years": args.test_years,
                    "scenarios": scenario_results,
                },
            )

    store.write_json(
        "metrics/future_drift.json",
        {
            "compound_shift_held_out": True,
            "test_years": args.test_years,
            "scenarios": scenario_results,
        },
    )
    write_decisions(store.path / "logs" / "drift_routing_decisions.csv", all_decisions)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = open_dataset(args.data_root)
    loaders = build_temporal_loaders(dataset, args)
    router, optimizer, num_classes, uses_strong_profile = build_router(
        args.base_checkpoint, device, args.adaptation_profile
    )
    store = create_run_store(args, loaders, num_classes, device)
    start_epoch, best_score = restore_training_state(
        args, store, router.new, optimizer, device
    )
    best_score = adapt_updated_model(
        router,
        optimizer,
        loaders,
        store,
        args,
        device,
        uses_strong_profile,
        start_epoch,
        best_score,
    )

    best_state = torch.load(
        store.path / "models" / "best.pt",
        map_location=device,
        weights_only=True,
    )
    router.new.load_state_dict(best_state)
    policy = calibrate_router(
        router, loaders.calibration, device, feature_set=args.router_features
    )
    store.write_json("metrics/router_policy.json", policy)
    result, decisions = evaluate_stream(
        router, loaders.future, policy, device, loaders.year_column
    )
    store.write_json("metrics/future_test.json", result)
    write_decisions(store.path / "logs" / "routing_decisions.csv", decisions)
    store.log(
        {
            "event": "final",
            "selected_checkpoint": "models/best.pt",
            "selection_metric": f"selection_{args.calibration_year}_macro_f1",
            "best_score": best_score,
            "calibration_protocol": loaders.calibration_protocol,
            "future_routed_macro_f1": result["all"]["routed"]["macro_f1"],
        }
    )
    if args.evaluate_drift:
        evaluate_drift_scenarios(router, policy, dataset, loaders, store, args, device)
    print(
        json.dumps(
            {
                "run": str(store.path),
                "future": {
                    name: result["all"][name]["macro_f1"]
                    for name in ("old", "new", "routed")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
