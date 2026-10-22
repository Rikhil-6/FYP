"""Conventionally fine-tune a uniform-pooling ViT on the temporal adaptation window."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from fyp_drift.artifacts.run_store import RunStore
from fyp_drift.data.clear100 import balanced_indices, open_dataset
from fyp_drift.data.corruptions import SCENARIOS, drift_transform, scenario_components
from fyp_drift.evaluation.metrics import classification_metrics
from fyp_drift.models.vision_transformer import TemporalViT
from fyp_drift.training.temporal_router import (
    build_temporal_loaders,
    seed_everything,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a DINOv2 ViT with uniform patch pooling and classification "
            "loss only."
        )
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--limit-adapt", type=int)
    parser.add_argument("--limit-calibrate", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument(
        "--drift-scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=("clean", "realistic_mix"),
    )
    parser.add_argument("--resume-run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")

    # These fields reproduce the established long-horizon protocol.
    args.long_horizon = True
    args.adapt_year = 2010
    args.calibration_year = 2011
    args.test_years = (2012, 2013, 2014)
    args.adaptation_profile = "realistic"
    args.selection_fraction = 0.5
    args.reuse_calibration_for_selection = False
    return args


def validate_checkpoint(path):
    """Require a uniform-pooling model selected before the adaptation year."""
    config_path = path.resolve().parent.parent / "config.json"
    if not config_path.is_file():
        raise ValueError("Historical run config is missing beside the checkpoint")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("method") != "baseline" or config.get("cv_val_year") != 2009:
        raise ValueError(
            "Conventional fine-tuning requires a baseline checkpoint selected on "
            "2009 and trained only on earlier years"
        )


def build_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = state["classifier.weight"].shape[0]
    model = TemporalViT("baseline", classes).to(device)
    model.load_state_dict(state)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.backbone.encoder.layer[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    backbone = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 5e-5},
        ],
        weight_decay=0.01,
    )
    return model, optimizer, classes


@torch.no_grad()
def evaluate(model, loader, device, year_column):
    model.eval()
    truth = []
    predictions = []
    records = []
    for images, labels, metadata in tqdm(loader, desc="evaluate fine-tuned ViT", unit="batch"):
        predicted = model(images.to(device))[0].argmax(1).cpu()
        truth.extend(labels.tolist())
        predictions.extend(predicted.tolist())
        for label, prediction, fields in zip(labels, predicted, metadata):
            records.append(
                {
                    "true_class": int(label),
                    "predicted_class": int(prediction),
                    "year": 2002 + int(fields[year_column]),
                }
            )
    return classification_metrics(truth, predictions), records


def train(model, optimizer, loaders, store, args, device, start_epoch, best_score):
    metric_name = "selection_2011_macro_f1"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        started = time.perf_counter()
        progress = tqdm(
            loaders.adaptation,
            desc=f"conventional fine-tune epoch {epoch}/{args.epochs}",
            unit="batch",
        )
        for step, (images, labels, _) in enumerate(progress, 1):
            labels = labels.to(device)
            logits = model(images.to(device))[0]
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step % args.log_every == 0 or step == len(loaders.adaptation):
                store.log(
                    {
                        "event": "fine_tune_batch",
                        "epoch": epoch,
                        "batch": step,
                        "batches": len(loaders.adaptation),
                        "loss": round(loss.item(), 6),
                    }
                )

        selection, _ = evaluate(model, loaders.selection, device, loaders.year_column)
        score = selection["macro_f1"]
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": round(loss_sum / len(loaders.adaptation.dataset), 4),
            metric_name: score,
            "train_wall_seconds": round(elapsed, 4),
            "train_images_per_second": round(len(loaders.adaptation.dataset) / elapsed, 2),
        }
        store.log(row)
        store.save_model(model, "last", epoch, metric_name, score)
        if score > best_score:
            best_score = score
            store.save_model(model, "best", epoch, metric_name, score)
        store.save_training_checkpoint(model, optimizer, epoch, best_score)
        print(json.dumps(row), flush=True)
    return best_score


def evaluate_horizon(model, dataset, loaders, store, args, device):
    results = {}
    records = []
    for scenario in args.drift_scenarios:
        results[scenario] = {}
        for severity, year in enumerate(args.test_years, 1):
            subset = dataset.get_subset(
                loaders.test_split,
                transform=drift_transform(scenario, severity, args.seed),
            )
            indices = torch.where(
                subset.metadata_array[:, loaders.year_column] == year - 2002
            )[0].tolist()
            if args.limit_test:
                indices = balanced_indices(
                    subset.rows,
                    indices,
                    min(args.limit_test, len(indices)),
                    args.seed,
                )
            loader = DataLoader(Subset(subset, indices), batch_size=args.batch_size)
            metrics, rows = evaluate(model, loader, device, loaders.year_column)
            results[scenario][str(year)] = {
                "severity": 0 if scenario == "clean" else severity,
                "metrics": metrics,
            }
            components = "+".join(scenario_components(scenario, 0, severity, args.seed))
            reported_severity = 0 if scenario == "clean" else severity
            records.extend(
                {
                    "scenario": scenario,
                    "severity": reported_severity,
                    "components": components,
                    **row,
                }
                for row in rows
            )
            print(
                json.dumps(
                    {
                        "event": "fine_tune_evaluation",
                        "scenario": scenario,
                        "year": year,
                        "macro_f1": metrics["macro_f1"],
                    }
                ),
                flush=True,
            )
    store.write_json("metrics/future_drift.json", {"scenarios": results})
    with (store.path / "logs" / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return results


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    validate_checkpoint(args.base_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = open_dataset(args.data_root)
    loaders = build_temporal_loaders(dataset, args)
    model, optimizer, classes = build_model(args.base_checkpoint, device)

    if args.resume_run_dir:
        store = RunStore.from_existing(args.resume_run_dir)
        checkpoint = torch.load(
            store.path / "models" / "resume.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
    else:
        store = RunStore(args.output, "finetune", args.run_id)
        store.write_json(
            "config.json",
            vars(args)
            | {
                "method": "baseline",
                "dataset": "CLEAR-100",
                "num_classes": classes,
                "device": device,
                "objective": "classification_only",
                "pooling": "uniform_patch_mean_plus_class_token",
                "trainable_backbone_blocks": 3,
                "torch_version": torch.__version__,
                "cuda_build": torch.version.cuda,
            },
        )
        start_epoch = 1
        best_score = -1.0

    best_score = train(
        model, optimizer, loaders, store, args, device, start_epoch, best_score
    )
    model.load_state_dict(
        torch.load(store.path / "models" / "best.pt", map_location=device, weights_only=True)
    )
    results = evaluate_horizon(model, dataset, loaders, store, args, device)
    store.log(
        {
            "event": "final",
            "selected_checkpoint": "models/best.pt",
            "selection_metric": "selection_2011_macro_f1",
            "best_score": best_score,
        }
    )
    print(json.dumps({"run": str(store.path), "results": results}, indent=2))


if __name__ == "__main__":
    main()
