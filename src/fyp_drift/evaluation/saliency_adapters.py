"""Saliency adapters on a frozen DINOv2-L: does saliency help when it decides which patches are read?

The backbone stays frozen; frozen final patch tokens are cached once per condition. Small
heads are then adapted on the 2014-15 adaptation set, checkpointed on adapt-val, and scored
on the 2016-17 test half. Every head starts from the historical linear head.

Arms (differ only in how patch tokens are pooled before the linear head):
  mean            equal weights                                       control
  evidence        top-40% patches by the frozen model's evidence for  pure saliency
                  its own predicted class (no labels at test time)
  shuffled        the same number of random patches                   control for evidence
  attention       learned per-patch weights (small MLP + softmax)     plain saliency adapter
  guided          attention plus a KL pull toward the frozen evidence saliency-guided adapter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_CACHE, DEFAULT_MANIFEST, SALIENT_FRACTION, Backbone, cluster_bootstrap, image_cache, image_loader,
    load_rows, log, scores, study_splits, write_json,
)

ARMS = ("mean", "evidence", "shuffled", "attention", "guided")
SPLITS = ("adapt_train", "adapt_val", "future_test")


def token_cache(backbone, cache, rows, path, cloud_alpha, workers):
    """fp16 memmap of frozen final patch tokens, shape (images, patches, width)."""
    path = Path(path)
    done = path.with_suffix(".done")
    if done.exists():
        return np.load(path, mmap_mode="r")
    tokens = None
    start = 0
    with torch.no_grad():
        for images, _, _ in image_loader(cache, rows, 64, workers, cloud_alpha):
            with torch.autocast("cuda", dtype=torch.float16):
                batch = backbone(images.cuda(non_blocking=True))[:, 1:].half().cpu().numpy()
            if tokens is None:
                tokens = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(rows),) + batch.shape[1:])
            tokens[start:start + len(batch)] = batch
            start += len(batch)
    tokens.flush()
    done.write_text("ok")
    return np.load(path, mmap_mode="r")


class PooledHead(nn.Module):
    """Linear head over a weighted patch average; the weighting is the arm."""

    def __init__(self, arm, head_state, width, classes):
        super().__init__()
        self.arm = arm
        self.frozen = nn.Linear(width, classes)
        self.frozen.load_state_dict(head_state)
        self.frozen.requires_grad_(False)
        self.classifier = nn.Linear(width, classes)
        self.classifier.load_state_dict(head_state)
        self.scorer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1)) if arm in ("attention", "guided") else None

    def evidence(self, tokens):
        """Frozen-model evidence of each patch for the frozen model's own predicted class."""
        with torch.no_grad():
            predicted = self.frozen(tokens.mean(1)).argmax(1)
            return (tokens * self.frozen.weight[predicted][:, None, :]).sum(-1)

    def weights(self, tokens, generator=None):
        patches = tokens.shape[1]
        if self.arm == "mean":
            return torch.full(tokens.shape[:2], 1.0 / patches, device=tokens.device)
        if self.scorer is not None:
            return self.scorer(tokens).squeeze(-1).softmax(1)
        salient = round(SALIENT_FRACTION * patches)
        weights = torch.zeros(tokens.shape[:2], device=tokens.device).scatter_(1, self.evidence(tokens).topk(salient, 1).indices, 1.0 / salient)
        if self.arm == "shuffled":
            weights = weights.gather(1, torch.rand(weights.shape, generator=generator).argsort(1).to(weights.device))
        return weights

    def forward(self, tokens, generator=None):
        weights = self.weights(tokens, generator)
        return self.classifier((weights[..., None] * tokens).sum(1)), weights


def guidance_loss(model, tokens, weights):
    """KL(frozen-evidence distribution || learned weights), evidence standardised per image."""
    evidence = model.evidence(tokens)
    target = ((evidence - evidence.mean(1, keepdim=True)) / evidence.std(1, keepdim=True).clamp_min(1e-6)).softmax(1)
    return F.kl_div(weights.clamp_min(1e-8).log(), target, reduction="batchmean")


def batches(tokens, labels, batch_size, shuffle_seed=None):
    order = np.random.default_rng(shuffle_seed).permutation(len(labels)) if shuffle_seed is not None else np.arange(len(labels))
    for start in range(0, len(order), batch_size):
        chosen = np.sort(order[start:start + batch_size])
        yield torch.from_numpy(np.asarray(tokens[chosen])).cuda().float(), torch.from_numpy(labels[chosen]).cuda()


@torch.no_grad()
def predict(model, tokens, labels, generator):
    model.eval()
    return np.concatenate([model(x, generator)[0].argmax(1).cpu().numpy() for x, _ in batches(tokens, labels, 256)])


def train_arm(arm, seed, head_state, width, classes, data, args):
    torch.manual_seed(seed)
    model = PooledHead(arm, head_state, width, classes).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    best = {"val_f1": -1.0}
    for epoch in range(args.epochs):
        model.train()
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        for tokens, targets in batches(*data["adapt_train"], args.batch, shuffle_seed=seed * 1000 + epoch):
            logits, weights = model(tokens, generator)
            loss = F.cross_entropy(logits, targets)
            if arm == "guided":
                loss = loss + args.guide_weight * guidance_loss(model, tokens, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_f1 = scores(data["adapt_val"][1], predict(model, *data["adapt_val"], torch.Generator().manual_seed(1)), classes)["macro_f1"]
        if val_f1 > best["val_f1"]:
            best = {"val_f1": val_f1, "epoch": epoch + 1, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    predictions = predict(model, *data["future_test"], torch.Generator().manual_seed(2))
    return {"arm": arm, "seed": seed, "best_epoch": best["epoch"], "adapt_val_macro_f1": best["val_f1"],
            "future_test": scores(data["future_test"][1], predictions, classes), "predictions": predictions.tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("/mnt/sdd/FYP/runs/today/main"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/sdd/FYP/runs/today/saliency_adapters"))
    parser.add_argument("--tokens", type=Path, default=Path("/mnt/sdd/FYP/cache/adapter_tokens"))
    parser.add_argument("--backbone", default="facebook/dinov2-large")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--guide-weight", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    args.tokens.mkdir(parents=True, exist_ok=True)
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    cache = image_cache(rows, DEFAULT_CACHE, args.workers)
    head_state = torch.load(args.run / "head.pt", weights_only=True)
    cloud_alpha = json.loads((args.run / "calib.json").read_text())["selected_alpha"]
    width = head_state["weight"].shape[1]
    groups = np.array([r["group"] for r in splits["future_test"]])
    backbone = None
    results = []
    for condition, alpha in (("clean", 0.0), ("cloud", cloud_alpha)):
        data = {}
        for split in SPLITS:
            path = args.tokens / f"{condition}_{split}.npy"
            if not path.with_suffix(".done").exists() and backbone is None:
                backbone = Backbone(args.backbone).cuda().eval()
            data[split] = (token_cache(backbone, cache, splits[split], path, alpha, args.workers), np.array([r["class_id"] for r in splits[split]]))
            log("tokens ready", condition, split)
        for seed in args.seeds:
            for arm in ARMS:
                path = args.output / f"{condition}_{arm}_s{seed}.json"
                if path.exists():
                    results.append(json.loads(path.read_text()))
                    continue
                result = {**train_arm(arm, seed, head_state, width, classes, data, args), "condition": condition}
                write_json(path, result)
                results.append(result)
                log(f"{condition} {arm} s{seed}: val={result['adapt_val_macro_f1']:.4f} test={result['future_test']['macro_f1']:.4f}")
    labels = np.array([r["class_id"] for r in splits["future_test"]])
    summary = {"means": {}, "paired": []}
    for condition in ("clean", "cloud"):
        for arm in ARMS:
            f1 = [r["future_test"]["macro_f1"] for r in results if r["condition"] == condition and r["arm"] == arm]
            summary["means"][f"{condition}/{arm}"] = {"seeds": len(f1), "mean": float(np.mean(f1)), "sd": float(np.std(f1, ddof=1)) if len(f1) > 1 else None}
        for a, b in (("evidence", "shuffled"), ("evidence", "mean"), ("guided", "attention"), ("attention", "mean"), ("guided", "mean")):
            for seed in args.seeds:
                ra = next(r for r in results if (r["condition"], r["arm"], r["seed"]) == (condition, a, seed))
                rb = next(r for r in results if (r["condition"], r["arm"], r["seed"]) == (condition, b, seed))
                difference = (np.array(ra["predictions"]) == labels).astype(float) - (np.array(rb["predictions"]) == labels).astype(float)
                summary["paired"].append({"condition": condition, "a": a, "b": b, "seed": seed,
                                          "delta_macro_f1": ra["future_test"]["macro_f1"] - rb["future_test"]["macro_f1"],
                                          "delta_accuracy": cluster_bootstrap(difference, groups, repetitions=1000)})
    write_json(args.output / "summary.json", summary)
    log("summary", json.dumps(summary["means"], indent=1))


if __name__ == "__main__":
    main()
