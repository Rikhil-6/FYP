"""Train ViT variants with validation-selected checkpoints and structured logs."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import make_loaders
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
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    loaders, year_column = make_loaders(args.data_root, args.batch_size, seed=args.seed,
                                        limit_train=args.limit_train, limit_eval=args.limit_eval)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalViT(args.method).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.encoder.layer[-1].parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
    store = RunStore(args.output, args.method, args.run_id)
    store.write_json("config.json", vars(args) | {"device": device})
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels, _ in loaders["train"]:
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
        validation, _ = evaluate(model, loaders, year_column, device, splits=("id_val",))
        score = validation["id_val"]["all"]["macro_f1"]
        row = {"epoch": epoch, "train_loss": round(loss_sum / len(loaders["train"].dataset), 4),
               "id_val_accuracy": validation["id_val"]["all"]["accuracy"],
               "id_val_macro_f1": score, "id_val_macro_iou": validation["id_val"]["all"]["macro_iou"]}
        store.log(row)
        store.save_model(model, "last", epoch, "id_val_macro_f1", score)
        if score > best_score:
            best_score = score
            store.save_model(model, "best", epoch, "id_val_macro_f1", score)
        print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(store.path / "models" / "best.pt", map_location=device, weights_only=True))
    report, records = evaluate(model, loaders, year_column, device)
    store.write_json("metrics/final.json", report)
    with (store.path / "logs" / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    store.log({"event": "final", "selected_checkpoint": "models/best.pt",
               "selection_metric": "id_val_macro_f1", "best_score": best_score})
    print(json.dumps({"run": str(store.path), "best_id_val_macro_f1": best_score,
                      "future_test": {key: report["test"]["all"][key]
                                      for key in ("accuracy", "macro_f1", "macro_iou")}}, indent=2))


if __name__ == "__main__":
    main()
