"""Train the historical baseline or saliency-gated ViT classifier."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from fyp_drift.artifacts.run_store import RunStore
from fyp_drift.data.clear100 import NUM_CLASSES, make_loaders, open_dataset, year_loader
from fyp_drift.evaluation.classifier import evaluate
from fyp_drift.models.vision_transformer import TemporalViT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a DINOv2 classifier on chronological CLEAR-100 splits."
    )
    parser.add_argument("--method", choices=("baseline", "xai"), required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--faithfulness-weight", type=float, default=0.1)
    parser.add_argument(
        "--cv-val-year",
        type=int,
        help="Train on earlier years and validate on this train-source year.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(args):
    """Build the standard split or one fixed-history temporal backtest."""
    if not args.cv_val_year:
        return make_loaders(
            args.data_root,
            args.batch_size,
            seed=args.seed,
            limit_train=args.limit_train,
            limit_eval=args.limit_eval,
        )

    dataset = open_dataset(args.data_root)
    year_column = dataset.metadata_fields.index("year")
    train_metadata = dataset.get_subset("train").metadata_array
    available_years = sorted(
        {
            2002 + int(value)
            for value in train_metadata[:, year_column].tolist()
            if value >= 0
        }
    )
    history = [year for year in available_years if year < args.cv_val_year]
    if len(history) < 2 or args.cv_val_year not in available_years:
        raise ValueError(
            f"Need at least two earlier train years and validation year "
            f"{args.cv_val_year}; available: {available_years}"
        )
    train_loader, _ = year_loader(
        dataset,
        "train",
        history,
        args.batch_size,
        shuffle=True,
        limit=args.limit_train,
        seed=args.seed,
    )
    validation_loader, _ = year_loader(
        dataset,
        "train",
        (args.cv_val_year,),
        args.batch_size,
        limit=args.limit_eval,
    )
    return {"train": train_loader, "id_val": validation_loader}, year_column


def build_model(method, device):
    model = TemporalViT(method, num_classes=NUM_CLASSES).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.backbone.encoder.layer[-1].parameters():
        parameter.requires_grad = True
    return model


def train_epoch(model, loader, optimizer, store, args, epoch, device):
    model.train()
    loss_sum = 0.0
    if device == "cuda":
        torch.cuda.synchronize()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    wall_start = time.perf_counter()

    progress = tqdm(
        loader,
        desc=f"{args.method} epoch {epoch}/{args.epochs}",
        unit="batch",
    )
    for step, (images, labels, _) in enumerate(progress, 1):
        images = images.to(device)
        labels = labels.to(device)
        logits, _, inverse_logits = model(
            images, return_counterfactual=args.method == "xai"
        )
        loss = F.cross_entropy(logits, labels)
        if inverse_logits is not None:
            selected_score = logits.gather(1, labels[:, None])
            inverse_score = inverse_logits.gather(1, labels[:, None])
            evidence_margin = selected_score - inverse_score
            loss += args.faithfulness_weight * F.relu(0.2 - evidence_margin).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if step % args.log_every == 0 or step == len(loader):
            store.log(
                {
                    "event": "train_batch",
                    "epoch": epoch,
                    "batch": step,
                    "batches": len(loader),
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
    timing = {
        "train_wall_seconds": round(wall_seconds, 4),
        "train_gpu_seconds": round(gpu_seconds, 4) if gpu_seconds is not None else None,
        "train_images_per_second": round(len(loader.dataset) / wall_seconds, 2),
    }
    return loss_sum / len(loader.dataset), timing


def write_predictions(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    loaders, year_column = build_loaders(args)
    if not len(loaders["train"].dataset) or not len(loaders["id_val"].dataset):
        raise ValueError("Training or validation split is empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.method, device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=2e-5,
    )
    store = RunStore(args.output, args.method, args.run_id)
    store.write_json(
        "config.json",
        vars(args)
        | {
            "dataset": "CLEAR-100",
            "num_classes": NUM_CLASSES,
            "device": device,
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
        },
    )

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_timing = train_epoch(
            model, loaders["train"], optimizer, store, args, epoch, device
        )
        validation, _ = evaluate(
            model, loaders, year_column, device, splits=("id_val",)
        )
        score = validation["id_val"]["all"]["macro_f1"]
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "id_val_accuracy": validation["id_val"]["all"]["accuracy"],
            "id_val_macro_f1": score,
            "id_val_macro_iou": validation["id_val"]["all"]["macro_iou"],
            "timing": train_timing | {"validation": validation["id_val"]["_timing"]},
        }
        store.log(row)
        store.save_model(model, "last", epoch, "id_val_macro_f1", score)
        if score > best_score:
            best_score = score
            store.save_model(model, "best", epoch, "id_val_macro_f1", score)
        print(json.dumps(row), flush=True)

    best_state = torch.load(
        store.path / "models" / "best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state)
    final_splits = (
        ("id_val",)
        if args.cv_val_year
        else (
            "id_val",
            "id_test",
            "val",
            "test",
        )
    )
    report, records = evaluate(model, loaders, year_column, device, splits=final_splits)
    store.write_json("metrics/final.json", report)
    write_predictions(store.path / "logs" / "predictions.csv", records)
    store.log(
        {
            "event": "final",
            "selected_checkpoint": "models/best.pt",
            "selection_metric": "id_val_macro_f1",
            "best_score": best_score,
        }
    )

    summary_split = "id_val" if args.cv_val_year else "test"
    print(
        json.dumps(
            {
                "run": str(store.path),
                "best_id_val_macro_f1": best_score,
                "reported_split": summary_split,
                "scores": {
                    key: report[summary_split]["all"][key]
                    for key in ("accuracy", "macro_f1", "macro_iou")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
