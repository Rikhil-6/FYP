"""LoRA + soft saliency gate trained end to end; the old classifier stays frozen, as in plain LoRA.

Saliency is the frozen head's standardised evidence on the unadapted tokens.
Example::  python -m fyp_drift.evaluation.fmow_v2.hybrid --seed 43
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from fyp_drift.evaluation import fmow_dinov2l_study as study
from fyp_drift.evaluation.fmow_v2.context import (
    BACKBONE,
    WIDTH,
    load_context,
    pool_2x2,
    trainable,
)
from torch import nn

BATCH, ACCUMULATE, EVAL_BATCH, WORKERS = 16, 2, 64, 4


class GatedPooling(nn.Module):
    """Attention pooling over 10 x 10 cells, optionally soft-gated by saliency, then the frozen old classifier."""

    def __init__(self, historical: nn.Linear, kind: str = "soft_gated", width: int = 8):
        super().__init__()
        self.kind = kind
        self.classifier = nn.Linear(WIDTH, historical.out_features)
        self.classifier.load_state_dict(historical.state_dict())
        self.classifier.requires_grad_(False)
        self.scorer = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, width), nn.GELU(), nn.Linear(width, 1))
        if kind == "soft_gated":
            self.gamma = nn.Parameter(torch.tensor(2.0))  # slope of the gate
            self.delta = nn.Parameter(torch.tensor(0.0))  # offset of the gate
            self.rho_logit = nn.Parameter(torch.tensor(0.0))  # floor rho = sigmoid(rho_logit), starts at 0.5

    def weights(self, cells, saliency):
        w = self.scorer(cells).squeeze(-1).softmax(1)
        if self.kind == "soft_gated":
            rho = torch.sigmoid(self.rho_logit)
            w = w * (rho + (1 - rho) * torch.sigmoid(self.gamma * saliency + self.delta))
            w = w / w.sum(1, keepdim=True).clamp_min(1e-8)
        return w

    def forward(self, cells, saliency):
        w = self.weights(cells, saliency)
        return self.classifier((w[..., None] * cells).sum(1)), w


class Hybrid(nn.Module):
    def __init__(self, historical, rank, blocks, kind):
        super().__init__()
        self.historical = historical
        self.backbone = study.Backbone(BACKBONE, blocks, rank)
        self.pooling = GatedPooling(historical, kind)

    @torch.no_grad()
    def saliency(self, early):
        """Old model's standardised evidence for its own prediction, on the unadapted tokens."""
        self.backbone.use_adapters(False)
        reference = pool_2x2(self.backbone.late_features(early)[:, 1:].float())
        self.backbone.use_adapters(True)
        predicted = self.historical(reference.mean(1)).argmax(1)
        evidence = (reference * self.historical.weight[predicted][:, None, :]).sum(-1)
        return (evidence - evidence.mean(1, keepdim=True)) / evidence.std(1, keepdim=True).clamp_min(1e-6)

    def forward(self, images):
        with torch.autocast("cuda", dtype=torch.float16):
            early = self.backbone.early_features(images)
            cells = pool_2x2(self.backbone.late_features(early)[:, 1:].float())
        z = self.saliency(early) if self.pooling.kind != "attention" else None
        logits, w = self.pooling(cells.float(), z)
        return logits, w, cells


@torch.no_grad()
def evaluate(model, ctx, rows, strength, region=None, explain=False):
    """Macro-F1, predictions and (optionally) each cell's exact contribution to the predicted class."""
    model.eval()
    predictions, contributions, weights = [], [], []
    for images, _, _ in study.image_loader(ctx.cache, rows, EVAL_BATCH, WORKERS, strength, cloud_region=region):
        logits, w, cells = model(images.cuda(non_blocking=True))
        p = logits.argmax(1)
        predictions.append(p.cpu())
        if explain:  # logit_c = sum_i w_i * (W_c . h_i) + b_c, so w_i * (W_c . h_i) is cell i's exact share
            evidence = cells.float() @ model.pooling.classifier.weight.T
            idx = torch.arange(len(p), device=p.device)
            contributions.append((w * (evidence[idx, :, p] - evidence.mean(2))).cpu())
            weights.append(w.cpu())
    predictions = torch.cat(predictions).numpy()
    labels = np.array([r["class_id"] for r in rows])
    explained = (torch.cat(contributions).numpy(), torch.cat(weights).numpy()) if explain else (None, None)
    return study.scores(labels, predictions, ctx.classes), predictions, *explained


def box_mask(rows, grid=10):
    """(images, grid*grid) boolean: cell centre inside the labelled object box."""
    centres = (np.arange(grid) + 0.5) / grid
    xs, ys = np.meshgrid(centres, centres)
    return np.stack(
        [
            ((np.abs(xs - r["box_cx"]) <= r["box_w"] / 2) & (np.abs(ys - r["box_cy"]) <= r["box_h"] / 2)).ravel()
            for r in rows
        ]
    )


def train_and_score(ctx, kind, rank, blocks, seed, max_epochs=5, patience=2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Hybrid(ctx.historical_head(), rank, blocks, kind).cuda()
    lora, pooling = trainable(model.backbone), trainable(model.pooling)
    optimizer = torch.optim.AdamW(
        [{"params": lora, "lr": 3e-4, "weight_decay": 0.01}, {"params": pooling, "lr": 1e-3, "weight_decay": 1e-4}]
    )
    scaler = torch.amp.GradScaler("cuda")

    best_val, best_state, best_epoch, stale, history, start = -1.0, None, 0, 0, [], time.time()
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (images, targets, _) in enumerate(
            study.image_loader(ctx.cache, ctx.splits["adapt_train"], BATCH, WORKERS, 0.0, shuffle_seed=seed + epoch)
        ):
            logits, _, _ = model(images.cuda(non_blocking=True))
            scaler.scale(F.cross_entropy(logits, targets.cuda()) / ACCUMULATE).backward()
            if (step + 1) % ACCUMULATE == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        val = evaluate(model, ctx, ctx.splits["adapt_val"], 0.0)[0]["macro_f1"]
        history.append({"epoch": epoch + 1, "val_f1": val, "seconds": time.time() - start})
        study.log(f"hybrid {kind} r{rank} b{blocks} s{seed} epoch {epoch + 1}: val={val:.4f}")
        if val > best_val:
            best_val, best_epoch, stale = val, epoch + 1, 0
            best_state = (
                [p.detach().clone() for p in lora],
                {k: v.clone() for k, v in model.pooling.state_dict().items()},
            )
        else:
            stale += 1
            if stale >= patience:
                break
    for p, kept in zip(lora, best_state[0], strict=True):
        p.data.copy_(kept)
    model.pooling.load_state_dict(best_state[1])
    train_seconds = time.time() - start

    rows = ctx.splits["future_test"]
    tests = {}
    for name, (strength, region) in ctx.test_conditions().items():
        metrics, preds, contrib, _ = evaluate(model, ctx, rows, strength, region, explain=(name == "clean"))
        tests[name] = {"future_test": metrics, "predictions": preds.tolist()}
        if contrib is not None:
            positive = np.clip(contrib, 0, None)
            tests[name]["box_share_of_support"] = float(
                ((positive * box_mask(rows)).sum(1) / (positive.sum(1) + 1e-9)).mean()
            )
    scalars = {k: float(v.detach()) for k, v in model.pooling.named_parameters() if v.ndim == 0}
    return dict(
        kind=kind,
        rank=rank,
        blocks=blocks,
        seed=seed,
        best_epoch=best_epoch,
        best_val=best_val,
        history=history,
        train_seconds=train_seconds,
        trainable_parameters=int(sum(p.numel() for p in lora + pooling)),
        scalars=scalars,
        tests=tests,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("soft_gated", "attention"), default="soft_gated")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="cap every split (smoke test only)")
    args = parser.parse_args(argv)
    ctx = load_context(args.limit)
    out = ctx.runs / "hybrid"
    out.mkdir(exist_ok=True)
    path = out / f"clean_{args.kind}_r{args.rank}_w8_b{args.blocks}_s{args.seed}.json"
    if path.exists() and not args.limit:
        study.log("exists, skipping", path.name)
        return
    result = train_and_score(ctx, args.kind, args.rank, args.blocks, args.seed)
    if not args.limit:
        study.write_json(path, result)
    study.log("done", path.name, {k: round(v["future_test"]["macro_f1"], 4) for k, v in result["tests"].items()})


if __name__ == "__main__":
    main()
