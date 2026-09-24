"""Train ViT variants with validation-selected checkpoints and structured logs."""
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

from data import NUM_CLASSES, make_loaders, open_dataset, year_loader
from evaluate import evaluate
from model import TemporalViT
from run_store import RunStore


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--cv-val-year", type=int, help="Train on earlier years and validate on this year within the official train split")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cv_val_year:
        dataset = open_dataset(args.data_root)
        year_column = dataset.metadata_fields.index("year")
        available = sorted({2002 + int(value) for value in dataset.get_subset("train").metadata_array[:, year_column].tolist() if value >= 0})
        history = [year for year in available if year < args.cv_val_year]
        if len(history) < 2 or args.cv_val_year not in available:
            parser.error(f"Need at least two earlier train years and validation year {args.cv_val_year}; available: {available}")
        train_loader, _ = year_loader(dataset, "train", history, args.batch_size,
                                      shuffle=True, limit=args.limit_train, seed=args.seed)
        val_loader, _ = year_loader(dataset, "train", (args.cv_val_year,), args.batch_size,
                                    limit=args.limit_eval)
        loaders = {"train": train_loader, "id_val": val_loader}
    else:
        loaders, year_column = make_loaders(args.data_root, args.batch_size, seed=args.seed,
                                            limit_train=args.limit_train, limit_eval=args.limit_eval)
    if not len(loaders["train"].dataset) or not len(loaders["id_val"].dataset):
        parser.error("Training or validation split is empty")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalViT(args.method, num_classes=NUM_CLASSES).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.encoder.layer[-1].parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
    store = RunStore(args.output, args.method, args.run_id)
    store.write_json("config.json", vars(args) | {"dataset": "CLEAR-100", "num_classes": NUM_CLASSES,
                                                  "device": device, "torch_version": torch.__version__,
                                                  "cuda_build": torch.version.cuda})
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        if device == "cuda":
            torch.cuda.synchronize()
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        wall_start = time.perf_counter()
        progress = tqdm(loaders["train"], desc=f"{args.method} epoch {epoch}/{args.epochs}", unit="batch")
        for step, (images, labels, _) in enumerate(progress, 1):
            images, labels = images.to(device), labels.to(device)
            logits, _, inverse_logits = model(images, return_counterfactual=args.method == "xai")
            loss = F.cross_entropy(logits, labels)
            if inverse_logits is not None:
                selected_score = logits.gather(1, labels[:, None])
                inverse_score = inverse_logits.gather(1, labels[:, None])
                loss = loss + args.faithfulness_weight * F.relu(0.2 - (selected_score - inverse_score)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step % args.log_every == 0 or step == len(loaders["train"]):
                store.log({"event": "train_batch", "epoch": epoch, "batch": step,
                           "batches": len(loaders["train"]), "loss": round(loss.item(), 6)})
        if device == "cuda":
            gpu_end.record()
            torch.cuda.synchronize()
            gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000
        else:
            gpu_seconds = None
        wall_seconds = time.perf_counter() - wall_start
        validation, _ = evaluate(model, loaders, year_column, device, splits=("id_val",))
        score = validation["id_val"]["all"]["macro_f1"]
        row = {"epoch": epoch, "train_loss": round(loss_sum / len(loaders["train"].dataset), 4),
               "id_val_accuracy": validation["id_val"]["all"]["accuracy"],
               "id_val_macro_f1": score, "id_val_macro_iou": validation["id_val"]["all"]["macro_iou"]}
        row["timing"] = {"train_wall_seconds": round(wall_seconds, 4),
                         "train_gpu_seconds": round(gpu_seconds, 4) if gpu_seconds is not None else None,
                         "train_images_per_second": round(len(loaders["train"].dataset) / wall_seconds, 2),
                         "validation": validation["id_val"]["_timing"]}
        store.log(row)
        store.save_model(model, "last", epoch, "id_val_macro_f1", score)
        if score > best_score:
            best_score = score
            store.save_model(model, "best", epoch, "id_val_macro_f1", score)
        print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(store.path / "models" / "best.pt", map_location=device, weights_only=True))
    report, records = evaluate(model, loaders, year_column, device,
                               splits=("id_val",) if args.cv_val_year else ("id_val", "id_test", "val", "test"))
    store.write_json("metrics/final.json", report)
    with (store.path / "logs" / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    store.log({"event": "final", "selected_checkpoint": "models/best.pt",
               "selection_metric": "id_val_macro_f1", "best_score": best_score})
    summary_split = "id_val" if args.cv_val_year else "test"
    print(json.dumps({"run": str(store.path), "best_id_val_macro_f1": best_score,
                      "reported_split": summary_split,
                      "scores": {key: report[summary_split]["all"][key]
                                 for key in ("accuracy", "macro_f1", "macro_iou")}}, indent=2))


if __name__ == "__main__":
    main()
