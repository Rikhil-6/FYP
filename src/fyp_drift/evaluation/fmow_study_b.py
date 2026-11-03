"""Matched supervised FMoW adaptation with plain and saliency anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Dinov2Model

from fyp_drift.data.clear100 import MEAN, STD
from fyp_drift.evaluation.fmow_layer_selection import add_group_lora
from fyp_drift.evaluation.fmow_saliency_layer_pilot import divide_groups, evaluate
from fyp_drift.evaluation.fmow_temporal_pilot import FMoWImages, load_manifest
from fyp_drift.models.vision_transformer import BACKBONE


class StudyImages(Dataset):
    def __init__(self, rows, shift, alpha, seed):
        self.base = FMoWImages(rows)
        self.rows = rows
        self.shift = shift
        self.alpha = alpha
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label, _ = self.base[index]
        if self.shift == "cloud":
            image = cloud_image(image, self.rows[index]["local_path"], self.alpha, self.seed)
        return image, label, index


def cloud_image(image, path, alpha, seed):
    """Fixed localized near-white overlay; same path gets the same overlay."""
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
    noise = torch.rand((1, 1, 16, 16), generator=generator)
    smooth = F.avg_pool2d(noise, 5, stride=1, padding=2)
    mask = (smooth >= torch.quantile(smooth, 0.8)).float()
    mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)[0]
    white = torch.tensor([(1 - mean) / std for mean, std in zip(MEAN, STD)])[:, None, None]
    return image * (1 - alpha * mask) + white * (alpha * mask)


def data_loader(rows, batch_size, workers, shift, alpha, seed, shuffle=False, epoch=0):
    return DataLoader(StudyImages(rows, shift, alpha, seed), batch_size=batch_size,
                      shuffle=shuffle, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed + epoch),
                      pin_memory=True)


def anchor_weights(reference_tokens, head, mode, seed):
    """All modes sum to one; saliency and shuffled select equal patch counts."""
    predicted = head(reference_tokens.mean(1)).argmax(1)
    evidence = (reference_tokens * head.weight[predicted, None, :]).sum(-1)
    count = round(0.4 * evidence.shape[1])
    selected = evidence.topk(count, dim=1).indices
    weights = torch.zeros_like(evidence).scatter_(1, selected, 1 / count)
    if mode == "uniform":
        weights.fill_(1 / evidence.shape[1])
    elif mode == "shuffled":
        order = torch.rand(evidence.shape, generator=torch.Generator().manual_seed(seed)).argsort(1)
        weights = weights.gather(1, order.to(weights.device))
    elif mode != "saliency":
        raise ValueError(mode)
    return weights


def fit(model, reference, head, rows, val_rows, args, mode, alpha):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses, anchors, entropies, marginals = [], [], [], []
        batches = data_loader(rows, args.batch_size, args.workers, args.shift, alpha,
                              args.seed, shuffle=True, epoch=epoch)
        for batch_id, (images, labels, _) in enumerate(batches):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad(), torch.autocast("cuda"):
                old = reference(pixel_values=images).last_hidden_state[:, 1:].float()
                if mode != "plain":
                    weights = anchor_weights(old, head, mode,
                                             args.seed + epoch * 100000 + batch_id)
            with torch.autocast("cuda"):
                new = model(pixel_values=images).last_hidden_state[:, 1:].float()
                logits = head(new.mean(1))
                cross_entropy = F.cross_entropy(logits, labels)
                anchor = (weights * (new - old).square().mean(-1)).sum(1).mean() if mode != "plain" else 0
                loss = cross_entropy + args.anchor_weight * anchor
                probabilities = logits.softmax(1)
                entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1).mean()
                mean_prob = probabilities.mean(0)
                marginal = -(mean_prob * mean_prob.clamp_min(1e-8).log()).sum()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(cross_entropy.detach()))
            anchors.append(float(anchor.detach()) if mode != "plain" else 0.0)
            entropies.append(float(entropy.detach()))
            marginals.append(float(marginal.detach()))
        val, _ = evaluate(model, head, data_loader(val_rows, args.batch_size, args.workers,
                                                  args.shift, alpha, args.seed), "cuda")
        point = {"epoch": epoch + 1, "cross_entropy": float(np.mean(losses)),
                 "anchor_mse": float(np.mean(anchors)),
                 "prediction_entropy": float(np.mean(entropies)),
                 "marginal_entropy": float(np.mean(marginals)),
                 "adapt_val_macro_f1": val["macro_f1"]}
        history.append(point)
        print(mode, point, flush=True)
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group", choices=("E", "M", "L"), default="E")
    parser.add_argument("--shift", choices=("clean", "cloud"), default="clean")
    parser.add_argument("--cloud-alpha", type=float,
                        help="fixed value; omit to calibrate on historical validation")
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=0, help="bounded execution check")
    parser.add_argument("--limit-val", type=int, default=0, help="bounded execution check")
    parser.add_argument("--limit-test", type=int, default=0, help="bounded execution check")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("GPU required")
    if (args.cloud_alpha is not None and not 0 <= args.cloud_alpha <= 1) or args.anchor_weight < 0:
        parser.error("invalid cloud alpha or anchor weight")
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, categories = load_manifest(args.manifest)
    adapt = sorted((r for r in rows if r["period"] == "adapt_2014_15"), key=lambda r: r["local_path"])
    future = sorted((r for r in rows if r["period"] == "future_2016_17"), key=lambda r: r["local_path"])
    historical = sorted((r for r in rows if r["period"] == "historical_2010_13"),
                        key=lambda r: r["local_path"])
    train, val = divide_groups(adapt, 0.1, args.seed)
    _, test = divide_groups(future, 0.5, args.seed + 1)
    if args.limit_train:
        train = train[:args.limit_train]
    if args.limit_val:
        val = val[:args.limit_val]
    if args.limit_test:
        test = test[:args.limit_test]
    head = nn.Linear(384, len(categories)).cuda().eval()
    head.load_state_dict(torch.load(args.head, map_location="cuda", weights_only=True))
    for parameter in head.parameters():
        parameter.requires_grad = False
    reference = Dinov2Model.from_pretrained(BACKBONE).cuda().eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    calibration = []
    alpha = 0.0
    if args.shift == "cloud":
        if args.cloud_alpha is None:
            _, hist_val = divide_groups(historical, 0.1, args.seed)
            for candidate in (0.0, 0.2, 0.35, 0.5, 0.7):
                score, _ = evaluate(reference, head,
                                    data_loader(hist_val, args.batch_size, args.workers,
                                                "cloud", candidate, args.seed), "cuda")
                calibration.append({"alpha": candidate, "macro_f1": score["macro_f1"]})
            clean = calibration[0]["macro_f1"]
            alpha = min(calibration[1:], key=lambda item:
                        abs((clean - item["macro_f1"]) - 0.075))["alpha"]
            selected_score = next(item["macro_f1"] for item in calibration if item["alpha"] == alpha)
            if not 0.05 <= clean - selected_score <= 0.10:
                raise RuntimeError("cloud candidates did not produce a 5–10 point historical-validation drop")
            print("cloud calibration", calibration, "selected", alpha, flush=True)
        else:
            alpha = args.cloud_alpha
    settings = {**vars(args), "manifest": str(args.manifest), "head": str(args.head),
                "output": str(args.output), "git_head": revision, "effective_cloud_alpha": alpha,
                "counts": {"adapt_train": len(train), "adapt_val": len(val), "future_test": len(test)},
                "cloud_calibration": calibration}
    config_path = args.output / "config.json"
    if config_path.exists() and not args.force:
        old_settings = json.loads(config_path.read_text())
        if old_settings != settings:
            raise RuntimeError("output directory contains a different configuration")
    config_path.write_text(json.dumps(settings, indent=2))
    test_data = data_loader(test, args.batch_size, args.workers, args.shift, alpha, args.seed)
    for mode in ("frozen", "plain", "uniform", "shuffled", "saliency"):
        path = args.output / f"{mode}.json"
        if path.exists() and not args.force:
            print("skip", path, flush=True)
            continue
        torch.manual_seed(args.seed)
        start = time.time()
        if mode == "frozen":
            model = reference
            count, history = 0, []
        else:
            model = Dinov2Model.from_pretrained(BACKBONE).cuda()
            count = add_group_lora(model, args.group)
            model.cuda()
            model.eval()
            images = next(iter(test_data))[0].cuda()
            with torch.inference_mode(), torch.autocast("cuda"):
                before = head(reference(pixel_values=images).last_hidden_state[:, 1:].mean(1).float())
                after = head(model(pixel_values=images).last_hidden_state[:, 1:].mean(1).float())
            if not torch.allclose(before, after, atol=1e-5, rtol=1e-5):
                raise RuntimeError(f"{mode} failed step-zero equivalence")
            history = fit(model, reference, head, train, val, args, mode, alpha)
        metrics, predictions = evaluate(model, head, test_data, "cuda")
        counts = np.bincount(predictions, minlength=len(categories))
        result = {"mode": mode, "seed": args.seed, "git_head": revision,
                  "trainable_parameters": count, "history": history,
                  "future_test": metrics, "prediction_histogram": counts.tolist(),
                  "predictions": predictions, "seconds": time.time() - start}
        path.write_text(json.dumps(result, indent=2))
        print(mode, {k: v for k, v in metrics.items() if k != "per_class_f1"}, flush=True)
        if mode != "frozen":
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
