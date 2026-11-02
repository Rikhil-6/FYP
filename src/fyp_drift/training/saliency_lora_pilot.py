"""Matched LoRA pilot of fixed-teacher saliency-weighted view consistency.

This is an exploratory mechanism experiment, separate from the frozen main
protocol. All arms see the same paired clean/shifted images and train the same
query/value LoRA modules and patch-only classifier. Only consistency weighting
differs. The historical teacher and its class-evidence masks stay frozen.
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
from torch.utils.data import DataLoader, Dataset

from fyp_drift.data.clear100 import open_dataset, year_loader, stratified_year_partition_loaders
from fyp_drift.data.corruptions import drift_transform, realistic_mixed_drift_transform
from fyp_drift.evaluation.metrics import classification_metrics
from fyp_drift.models.vision_transformer import TemporalViT


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        for parameter in base.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


class PairedDataset(Dataset):
    def __init__(self, clean, shifted):
        if clean.indices != shifted.indices:
            raise ValueError("Paired views must have identical manifest indices")
        self.clean, self.shifted = clean, shifted

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, label, _ = self.clean[index]
        shifted, other_label, _ = self.shifted[index]
        if label != other_label:
            raise ValueError("Paired views have different labels")
        return clean, shifted, label


def teacher_saliency(tokens, class_weights, labels, top_fraction=0.2):
    """Normalized top-patch evidence for the known training class."""
    scores = torch.einsum("bpd,bd->bp", tokens, class_weights[labels])
    count = max(1, round(scores.shape[1] * top_fraction))
    top = scores.topk(count, dim=1).indices
    weights = torch.zeros_like(scores).scatter_(1, top, 1.0 / count)
    return weights


def consistency_weights(mode, saliency):
    if mode == "uniform":
        return torch.full_like(saliency, 1.0 / saliency.shape[1])
    if mode == "saliency":
        return saliency
    if mode == "shuffled":
        return saliency[:, torch.randperm(saliency.shape[1], device=saliency.device)]
    raise ValueError(mode)


def build_model(checkpoint, device, rank):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = state["classifier.weight"].shape[0]
    model = TemporalViT("baseline", classes, pooling_mode="patch_only")
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.backbone.encoder.layer[-3:]:
        attention = block.attention.attention
        attention.query = LoRALinear(attention.query, rank=rank)
        attention.value = LoRALinear(attention.value, rank=rank)
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    truth, predictions, rows = [], [], []
    for images, labels, _ in loader:
        predicted = model(images.to(device))[0].argmax(1).cpu().tolist()
        truth.extend(labels.tolist())
        predictions.extend(predicted)
        ids = loader.dataset.indices[len(rows):len(rows) + len(predicted)]
        rows.extend(zip(ids, labels.tolist(), predicted))
    return classification_metrics(truth, predictions), rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--lora-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-lr", type=float, default=5e-5)
    parser.add_argument("--limit-adapt", type=int, default=600)
    parser.add_argument("--limit-select", type=int, default=400)
    parser.add_argument("--limit-test", type=int, default=500)
    parser.add_argument("--modes", nargs="+", choices=("none", "uniform", "shuffled", "saliency"),
                        default=("none", "uniform", "shuffled", "saliency"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output already exists")
    if args.epochs < 1 or args.batch_size < 1 or args.rank < 1:
        parser.error("epochs, batch size, and rank must be positive")
    if args.consistency_weight < 0 or args.lora_lr <= 0 or args.classifier_lr <= 0:
        parser.error("consistency weight must be nonnegative and rates positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This pilot requires a visible GPU")
    dataset = open_dataset(args.data_root)
    clean, _ = year_loader(dataset, "train_source", (2010,), args.batch_size,
                           limit=args.limit_adapt, seed=args.seed,
                           image_transform=drift_transform("clean", 1, args.seed))
    shifted, _ = year_loader(dataset, "train_source", (2010,), args.batch_size,
                             limit=args.limit_adapt, seed=args.seed,
                             image_transform=realistic_mixed_drift_transform((1, 2, 3), args.seed))
    paired = PairedDataset(clean.dataset, shifted.dataset)
    selection, _, _ = stratified_year_partition_loaders(
        dataset, "train_source", 2011, args.batch_size, fraction=0.5,
        limit=args.limit_select, seed=args.seed,
        image_transform=realistic_mixed_drift_transform(2, args.seed),
    )
    test_loaders = {
        name: year_loader(dataset, "test_source", (2014,), args.batch_size,
                          limit=args.limit_test, seed=args.seed,
                          image_transform=drift_transform(name, 3, args.seed))[0]
        for name in ("clean", "viewpoint_shift", "occlusion", "realistic_mix")
    }
    historical = build_model(args.base_checkpoint, device, args.rank)
    # Teacher uses the original uniform checkpoint; LoRA is zero at initialization.
    historical.eval()
    for parameter in historical.parameters():
        parameter.requires_grad = False
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps({
        **vars(args), "base_checkpoint": str(args.base_checkpoint), "device": device,
        "adaptation_year": 2010, "selection_year": 2011,
        "evaluation_year": 2014, "selection_protocol": "disjoint_class_stratified",
        "teacher_map": "frozen_true_class_top20_patch_contribution",
    }, indent=2, default=str))
    summary = {}
    for mode in args.modes:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = build_model(args.base_checkpoint, device, args.rank)
        lora_parameters = [p for name, p in model.named_parameters()
                           if p.requires_grad and (".down." in name or ".up." in name)]
        optimizer = torch.optim.AdamW([
            {"params": lora_parameters, "lr": args.lora_lr},
            {"params": model.classifier.parameters(), "lr": args.classifier_lr},
        ], weight_decay=0.01)
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(paired, batch_size=args.batch_size, shuffle=True,
                                  generator=generator)
        best_score = -1.0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses, classification_losses, consistency_losses = [], [], []
            for clean_images, shifted_images, labels in train_loader:
                clean_images = clean_images.to(device)
                shifted_images = shifted_images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    teacher_tokens = historical.backbone(
                        pixel_values=clean_images
                    ).last_hidden_state[:, 1:]
                    saliency = teacher_saliency(
                        teacher_tokens, historical.classifier.weight, labels
                    )
                clean_tokens = model.backbone(
                    pixel_values=clean_images
                ).last_hidden_state[:, 1:]
                shifted_tokens = model.backbone(
                    pixel_values=shifted_images
                ).last_hidden_state[:, 1:]
                clean_logits = model.classifier(clean_tokens.mean(1))
                shifted_logits = model.classifier(shifted_tokens.mean(1))
                classification_loss = (
                    F.cross_entropy(clean_logits, labels, label_smoothing=0.05)
                    + F.cross_entropy(shifted_logits, labels, label_smoothing=0.05)
                ) / 2
                loss = classification_loss
                consistency_loss = torch.zeros((), device=device)
                if mode != "none":
                    weights = consistency_weights(mode, saliency)
                    distance = (F.normalize(clean_tokens, dim=-1)
                                - F.normalize(shifted_tokens, dim=-1)).square().sum(-1)
                    consistency_loss = (weights * distance).sum(1).mean()
                    loss = loss + args.consistency_weight * consistency_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                classification_losses.append(float(classification_loss.detach()))
                consistency_losses.append(float(consistency_loss.detach()))
            selected, _ = evaluate(model, selection, device)
            history.append({"epoch": epoch, "mean_loss": float(np.mean(losses)),
                            "mean_classification_loss": float(np.mean(classification_losses)),
                            "mean_consistency_loss": float(np.mean(consistency_losses)),
                            "selection_macro_f1": selected["macro_f1"]})
            print(json.dumps({"mode": mode, **history[-1]}), flush=True)
            if selected["macro_f1"] > best_score:
                best_score = selected["macro_f1"]
                torch.save(model.state_dict(), args.output / f"{mode}.pt")
        model.load_state_dict(torch.load(args.output / f"{mode}.pt", map_location=device,
                                         weights_only=True))
        scores = {}
        with (args.output / f"{mode}-predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("scenario", "dataset_index", "true_class", "predicted_class"))
            for scenario, loader in test_loaders.items():
                metrics, rows = evaluate(model, loader, device)
                scores[scenario] = {"count": metrics["count"],
                                    "accuracy": metrics["accuracy"],
                                    "macro_f1": metrics["macro_f1"]}
                writer.writerows((scenario, *row) for row in rows)
        summary[mode] = {"history": history, "scores": scores,
                         "trainable_parameters": sum(p.numel() for p in model.parameters()
                                                     if p.requires_grad)}
        print(json.dumps({"mode": mode, "scores": scores}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
