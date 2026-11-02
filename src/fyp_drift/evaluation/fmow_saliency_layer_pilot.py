"""Small FMoW pilot: use patch evidence to choose which ViT blocks to adapt.

The diagnostic sees only historical images and unlabeled future_a. All E/M/L
arms use the same label-free objective. Future_test labels are opened only for
the final score. This is an architecture pilot, not the full study protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TVF
from tqdm import tqdm
from transformers import Dinov2Model

from fyp_drift.evaluation.fmow_layer_selection import add_group_lora, profile
from fyp_drift.evaluation.fmow_temporal_pilot import (
    FMoWImages, limit_rows, load_manifest,
)
from fyp_drift.data.clear100 import MEAN, STD
from fyp_drift.models.vision_transformer import BACKBONE


class UnlabeledImages(Dataset):
    """Keep future_a labels out of the diagnostic and adaptation loaders."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["local_path"]) as opened:
            opened.draft("RGB", (224, 224))
            image = opened.convert("RGB")
        return TVF.normalize(TVF.to_tensor(TVF.resize(image, (224, 224))), MEAN, STD)


def divide_groups(rows, test_fraction, seed):
    groups = [f"{row['country']}:{row['utm']}" for row in rows]
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                                         random_state=seed).split(rows, groups=groups))
    left, right = [rows[i] for i in train], [rows[i] for i in test]
    assert not ({groups[i] for i in train} & {groups[i] for i in test})
    return left, right


def loader(rows, batch_size, workers, labeled):
    dataset = FMoWImages(rows) if labeled else UnlabeledImages(rows)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def head_features(backbone, data, device):
    features, labels = [], []
    backbone.eval()
    for images, target, _ in tqdm(data, desc="head features", leave=False):
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            tokens = backbone(pixel_values=images.to(device)).last_hidden_state[:, 1:]
        features.append(tokens.mean(1).float().cpu())
        labels.append(target)
    return torch.cat(features), torch.cat(labels)


def fit_head(train_features, train_labels, val_features, val_labels, classes, seed):
    torch.manual_seed(seed)
    head = nn.Linear(train_features.shape[1], classes)
    counts = torch.bincount(train_labels, minlength=classes).float()
    weights = 1 / counts.clamp_min(1)
    weights /= weights[counts > 0].mean()
    weights = weights.clamp_max(10)
    weights[counts == 0] = 0
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, stale = -1.0, None, 0
    generator = torch.Generator().manual_seed(seed)
    for _ in range(30):
        head.train()
        for indices in torch.randperm(len(train_labels), generator=generator).split(256):
            logits = head(train_features[indices])
            loss = F.cross_entropy(logits, train_labels[indices], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            prediction = head(val_features).argmax(1).numpy()
        score = f1_score(val_labels.numpy(), prediction, labels=np.arange(classes),
                         average="macro", zero_division=0)
        if score > best:
            best, best_state, stale = score, copy.deepcopy(head.state_dict()), 0
        else:
            stale += 1
            if stale >= 5:
                break
    head.load_state_dict(best_state)
    return head, float(best)


@torch.inference_mode()
def diagnostic_features(backbone, head, data, device):
    blocks, scores = [], []
    backbone.eval()
    head.eval()
    for images in tqdm(data, desc="diagnostic features", leave=False):
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            output = backbone(pixel_values=images.to(device), output_hidden_states=True)
        final = output.last_hidden_state[:, 1:].float()
        logits = head(final.mean(1))
        predicted = logits.argmax(1)
        weights = head.weight.to(device)[predicted]
        evidence = torch.einsum("bpd,bd->bp", final, weights) / final.shape[1]
        reconstructed = evidence.sum(1) + head.bias.to(device)[predicted]
        chosen_logit = logits.gather(1, predicted[:, None]).squeeze(1)
        if not torch.allclose(reconstructed, chosen_logit, atol=1e-4, rtol=1e-4):
            raise AssertionError("patch evidence does not reconstruct the head logit")
        layer_tokens = torch.stack([state[:, 1:] for state in output.hidden_states[1:]], dim=1)
        blocks.append(layer_tokens.half().cpu().numpy())
        scores.append(evidence.cpu().numpy())
    return np.concatenate(blocks), np.concatenate(scores)


def adapt(backbone, head, data, device, epochs, lr):
    backbone.train()
    head.eval()
    optimizer = torch.optim.AdamW((p for p in backbone.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    history = []
    for epoch in range(epochs):
        losses = []
        for images in tqdm(data, desc=f"adapt epoch {epoch + 1}", leave=False):
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                tokens = backbone(pixel_values=images.to(device)).last_hidden_state[:, 1:]
                logits = head(tokens.mean(1).float())
                probabilities = logits.softmax(1)
                entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1).mean()
                mean_prob = probabilities.mean(0)
                diversity = (mean_prob * mean_prob.clamp_min(1e-8).log()).sum()
                loss = entropy + diversity
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    return history


@torch.inference_mode()
def evaluate(backbone, head, data, device):
    backbone.eval()
    head.eval()
    labels, predictions = [], []
    for images, target, _ in tqdm(data, desc="future_test", leave=False):
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            tokens = backbone(pixel_values=images.to(device)).last_hidden_state[:, 1:]
        logits = head(tokens.mean(1).float()).cpu()
        labels.extend(target.tolist())
        predictions.extend(logits.argmax(1).tolist())
    truth, predicted = np.asarray(labels), np.asarray(predictions)
    classes = np.arange(head.out_features)
    metrics = {"samples": len(truth),
               "accuracy": float(accuracy_score(truth, predicted)),
               "macro_f1": float(f1_score(truth, predicted, labels=classes,
                                          average="macro", zero_division=0)),
               "balanced_accuracy": float(recall_score(truth, predicted, labels=classes,
                                                        average="macro", zero_division=0)),
               "per_class_f1": f1_score(truth, predicted, labels=classes,
                                        average=None, zero_division=0).tolist()}
    return metrics, predictions


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/fmow-saliency-layer-pilot"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-period", type=int, default=0,
                        help="smoke-test limit; 0 uses the full manifest")
    parser.add_argument("--diagnostic-limit", type=int, default=256)
    parser.add_argument("--test-limit", type=int, default=0,
                        help="smoke-test limit; 0 uses all future_test images")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("this pilot requires a GPU")
    if args.batch_size < 1 or args.epochs < 1 or args.diagnostic_limit < 2:
        parser.error("invalid batch size, epochs or diagnostic limit")
    device = "cuda"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = {**vars(args), "manifest": str(args.manifest), "output": str(args.output),
              "git_head": revision, "backbone": BACKBONE}
    rows, categories = load_manifest(args.manifest)
    periods = {period: sorted((row for row in rows if row["period"] == period),
                              key=lambda row: row["local_path"])
               for period in ("historical_2010_13", "future_2016_17")}
    if args.limit_per_period:
        periods = {name: limit_rows(group, args.limit_per_period, args.seed)
                   for name, group in periods.items()}
    hist_train, hist_val = divide_groups(periods["historical_2010_13"], 0.1, args.seed)
    future_a, future_test = divide_groups(periods["future_2016_17"], 0.5, args.seed + 1)
    if args.test_limit:
        future_test = future_test[:args.test_limit]
    if min(map(len, (hist_train, hist_val, future_a, future_test))) < 2:
        parser.error("each split needs at least two images")
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "config.json", {**config, "split_counts": {
        "hist_train": len(hist_train), "hist_val": len(hist_val),
        "future_a": len(future_a), "future_test": len(future_test)}})

    backbone = Dinov2Model.from_pretrained(BACKBONE).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    head_path = args.output / "head.pt"
    if head_path.exists() and not args.force:
        head = nn.Linear(backbone.config.hidden_size, len(categories))
        head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    else:
        start = time.time()
        train_x, train_y = head_features(backbone, loader(hist_train, args.batch_size, args.workers, True), device)
        val_x, val_y = head_features(backbone, loader(hist_val, args.batch_size, args.workers, True), device)
        head, val_f1 = fit_head(train_x, train_y, val_x, val_y, len(categories), args.seed)
        torch.save(head.state_dict(), head_path)
        save_json(args.output / "head.json", {"seed": args.seed, "git_head": revision,
                   "validation_macro_f1": val_f1, "seconds": time.time() - start})
    head = head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad = False

    diagnostic_path = args.output / "diagnostic_pick.json"
    if diagnostic_path.exists() and not args.force:
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))["profile"]
    else:
        start = time.time()
        rng = np.random.default_rng(args.seed)
        hist_rows = [hist_train[i] for i in rng.choice(len(hist_train),
                     min(args.diagnostic_limit, len(hist_train)), replace=False)]
        target_rows = [future_a[i] for i in rng.choice(len(future_a),
                       min(args.diagnostic_limit, len(future_a)), replace=False)]
        hist_tokens, hist_scores = diagnostic_features(
            backbone, head, loader(hist_rows, args.batch_size, args.workers, False), device)
        target_tokens, target_scores = diagnostic_features(
            backbone, head, loader(target_rows, args.batch_size, args.workers, False), device)
        diagnostic = profile(hist_tokens, target_tokens, hist_scores, target_scores, args.seed)
        save_json(diagnostic_path, {"seed": args.seed, "git_head": revision,
                   "profile": diagnostic, "seconds": time.time() - start,
                   "source_images": len(hist_rows), "target_images": len(target_rows),
                   "head": "frozen historical patch-mean linear head"})
    print("Locked choices:", {name: entry["pick"] for name, entry in diagnostic.items()}, flush=True)
    del backbone
    torch.cuda.empty_cache()

    test_data = loader(future_test, args.batch_size, args.workers, True)
    target_data = loader(future_a, args.batch_size, args.workers, False)
    for group in ("A0", "E", "M", "L", "ALL"):
        output = args.output / f"{group}.json"
        if output.exists() and not args.force:
            print(f"Skipping existing {output}", flush=True)
            continue
        torch.manual_seed(args.seed)
        model = Dinov2Model.from_pretrained(BACKBONE).to(device)
        for parameter in model.parameters():
            parameter.requires_grad = False
        start = time.time()
        count, history = 0, []
        if group != "A0":
            count = add_group_lora(model, group)
            model.to(device)
            history = adapt(model, head, target_data, device, args.epochs, args.lr)
        metrics, predictions = evaluate(model, head, test_data, device)
        save_json(output, {"seed": args.seed, "git_head": revision, "config": config,
                   "group": group, "trainable_parameters": count, "history": history,
                   "future_test": metrics, "predictions": predictions,
                   "seconds": time.time() - start})
        print(group, metrics, flush=True)
        del model
        torch.cuda.empty_cache()
    counts = [json.loads((args.output / f"{name}.json").read_text(encoding="utf-8"))["trainable_parameters"]
              for name in ("E", "M", "L")]
    if len(set(counts)) != 1:
        raise AssertionError(f"E/M/L trainable parameter counts differ: {counts}")
    scores = {name: json.loads((args.output / f"{name}.json").read_text(encoding="utf-8"))["future_test"]
              for name in ("A0", "E", "M", "L", "ALL")}
    selected = {mode: {"group": item["pick"],
                       "macro_f1": scores[item["pick"]]["macro_f1"]}
                for mode, item in diagnostic.items()}
    save_json(args.output / "summary.json", {"seed": args.seed, "git_head": revision,
              "config": config, "diagnostic_choices": {name: item["pick"] for name, item in diagnostic.items()},
              "selected": selected, "scores": scores})


if __name__ == "__main__":
    main()
