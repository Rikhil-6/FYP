"""Chronological Yearbook pilot of saliency-guided masking during LoRA adaptation.

The historical head is trained through 1989. LoRA adapts on 1990-1999, selects
an epoch on 2000-2003, and evaluates once on 2004-2013. Saliency and random
masking preserve the same number of image patches. No future labels tune masks.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fyp_drift.evaluation.metrics import classification_metrics
from fyp_drift.models.vision_transformer import TemporalViT
from fyp_drift.training.saliency_lora_pilot import build_model, teacher_saliency


def mask_images(images, weights):
    side = round(weights.shape[1] ** 0.5)
    if side * side != weights.shape[1]:
        raise ValueError("patch weights require a square grid")
    keep = (weights > 0).reshape(-1, 1, side, side).to(images.dtype)
    keep = F.interpolate(keep, size=images.shape[-2:], mode="nearest")
    return images * keep  # zero is the normalized RGB channel mean


def make_loaders(args):
    from fyp_drift.evaluation.yearbook_replication import year_range, sampled_loader

    periods = {
        "historical_train": (1930, 1989, "train", args.historical_limit, args.seed),
        "historical_select": (1980, 1989, "test", args.selection_limit, args.seed + 1),
        "adaptation": (1990, 1999, "train", args.adapt_limit, args.seed + 2),
        "updated_select": (2000, 2003, "test", args.selection_limit, args.seed + 3),
        "future": (2004, 2013, "test", args.future_limit, args.seed + 4),
    }
    loaders = {}
    for name, (start, end, split, limit, seed) in periods.items():
        dataset = year_range(args.data_root, start, end, split)
        loaders[name] = sampled_loader(dataset, limit, seed, args.batch_size, 0)
    return loaders


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    truth, predictions, rows = [], [], []
    for images, labels, years in loader:
        predicted = model(images.to(device))[0].argmax(1).cpu().tolist()
        truth.extend(labels.tolist())
        predictions.extend(predicted)
        rows.extend((int(year), int(label), int(prediction)) for year, label, prediction
                    in zip(years.squeeze(-1).tolist(), labels.tolist(), predicted))
    metrics = classification_metrics(truth, predictions, classes=2)
    return {"count": metrics["count"], "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"]}, rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/wild-time"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--historical-limit", type=int, default=3000)
    parser.add_argument("--selection-limit", type=int, default=300)
    parser.add_argument("--adapt-limit", type=int, default=1000)
    parser.add_argument("--future-limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--historical-epochs", type=int, default=2)
    parser.add_argument("--adapt-epochs", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--keep-fraction", type=float, default=0.4)
    parser.add_argument("--mask-loss-weight", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    if not 0 < args.keep_fraction < 1 or args.mask_loss_weight < 0:
        parser.error("invalid mask parameters")
    return args


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Yearbook pilot requires a GPU")
    loaders = make_loaders(args)
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    historical = TemporalViT("baseline", 2, pooling_mode="patch_only").to(device)
    for parameter in historical.backbone.parameters():
        parameter.requires_grad = False
    for parameter in historical.saliency_head.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(historical.classifier.parameters(), lr=5e-4)
    best = -1.0
    historical_history = []
    for epoch in range(1, args.historical_epochs + 1):
        historical.train()
        historical.backbone.eval()
        for images, labels, _ in loaders["historical_train"]:
            logits = historical(images.to(device))[0]
            loss = F.cross_entropy(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score, _ = evaluate(historical, loaders["historical_select"], device)
        historical_history.append({"epoch": epoch, **score})
        print(json.dumps({"stage": "historical", **historical_history[-1]}), flush=True)
        if score["macro_f1"] > best:
            best = score["macro_f1"]
            torch.save(historical.state_dict(), args.output / "historical.pt")
    historical.load_state_dict(torch.load(args.output / "historical.pt",
                                          map_location=device, weights_only=True))
    historical.eval()
    for parameter in historical.parameters():
        parameter.requires_grad = False
    historical_future, historical_rows = evaluate(historical, loaders["future"], device)
    historical_selection, _ = evaluate(historical, loaders["updated_select"], device)

    summary = {"historical_history": historical_history,
               "historical_future": historical_future,
               "historical_2000_2003": historical_selection, "arms": {}}
    for mode in ("none", "random", "saliency"):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = build_model(args.output / "historical.pt", device, args.rank)
        lora_parameters = [p for name, p in model.named_parameters()
                           if p.requires_grad and (".down." in name or ".up." in name)]
        optimizer = torch.optim.AdamW([
            {"params": lora_parameters, "lr": 1e-3},
            {"params": model.classifier.parameters(), "lr": 5e-5},
        ], weight_decay=0.01)
        best = -1.0
        history = []
        for epoch in range(1, args.adapt_epochs + 1):
            model.train()
            losses = []
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(loaders["adaptation"].dataset), generator=generator)
            train_loader = DataLoader(
                torch.utils.data.Subset(loaders["adaptation"].dataset, order.tolist()),
                batch_size=args.batch_size,
            )
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)
                original_logits = model(images)[0]
                loss = F.cross_entropy(original_logits, labels)
                if mode != "none":
                    with torch.no_grad():
                        teacher_tokens = historical.backbone(
                            pixel_values=images).last_hidden_state[:, 1:]
                        predicted_class = historical.classifier(
                            teacher_tokens.mean(1)).argmax(1)
                        weights = teacher_saliency(
                            teacher_tokens, historical.classifier.weight,
                            predicted_class, args.keep_fraction,
                        )
                        if mode == "random":
                            weights = weights[:, torch.randperm(
                                weights.shape[1], device=device)]
                        masked_images = mask_images(images, weights)
                    loss = loss + args.mask_loss_weight * F.cross_entropy(
                        model(masked_images)[0], labels
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            score, _ = evaluate(model, loaders["updated_select"], device)
            history.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), **score})
            print(json.dumps({"mode": mode, **history[-1]}), flush=True)
            if score["macro_f1"] > best:
                best = score["macro_f1"]
                torch.save(model.state_dict(), args.output / f"{mode}.pt")
        model.load_state_dict(torch.load(args.output / f"{mode}.pt",
                                         map_location=device, weights_only=True))
        future, rows = evaluate(model, loaders["future"], device)
        summary["arms"][mode] = {"history": history, "future": future,
                                  "trainable_parameters": sum(p.numel() for p in model.parameters()
                                                              if p.requires_grad)}
        with (args.output / f"{mode}-future.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("year", "true_class", "predicted_class"))
            writer.writerows(rows)
        print(json.dumps({"mode": mode, "future": future}), flush=True)
    with (args.output / "historical-future.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("year", "true_class", "predicted_class"))
        writer.writerows(historical_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
