"""LoRA trained with the heads' protocol: best-validation epoch of up to 5, early stopping, all test conditions.

Learning rate 3e-4 was chosen once on rank-2 clean seed 42 from {1e-4, 3e-4}.
Example::  python -m fyp_drift.evaluation.fmow_v2.equal_lora --rank 2 --blocks 6 --condition clean --seed 43
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fyp_drift.evaluation import fmow_dinov2l_study as study
from fyp_drift.evaluation.fmow_v2.context import BACKBONE, load_context, trainable
from fyp_drift.evaluation.fmow_v2.models import adapter_state, save_model

BATCH, ACCUMULATE, EVAL_BATCH, WORKERS = 16, 2, 64, 4  # effective batch 32


def output_name(condition: str, rank: int, lr: float, blocks: int, seed: int) -> str:
    """File name used in ``runs/v2/lora_fair`` (six-block runs carry no block tag)."""
    tag = "" if blocks == 6 else f"_b{blocks}"
    return f"{condition}_r{rank}_lr{lr:g}{tag}_s{seed}.json"


def train_and_score(ctx, rank, blocks, condition, seed, lr, max_epochs=5, patience=2):
    head = ctx.historical_head()  # LoRA keeps the old classifier frozen and changes only the features
    alpha = ctx.overlay_alpha if condition == "cloud" else 0.0
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone = study.Backbone(BACKBONE, blocks, rank).cuda()
    params = trainable(backbone)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    best_val, best_params, best_epoch, stale, history, start = -1.0, None, 0, 0, [], time.time()
    for epoch in range(max_epochs):
        backbone.train()
        optimizer.zero_grad(set_to_none=True)
        loader = study.image_loader(
            ctx.cache, ctx.splits["adapt_train"], BATCH, WORKERS, alpha, shuffle_seed=seed + epoch
        )
        for step, (images, targets, _) in enumerate(loader):
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                tokens = backbone.late_features(backbone.early_features(images))[:, 1:].float()
                loss = F.cross_entropy(head(tokens.mean(1)), targets) / ACCUMULATE
            scaler.scale(loss).backward()
            if (step + 1) % ACCUMULATE == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        backbone.eval()
        val, _ = study.evaluate(
            backbone,
            head,
            study.image_loader(ctx.cache, ctx.splits["adapt_val"], EVAL_BATCH, WORKERS, alpha),
            ctx.classes,
        )
        history.append({"epoch": epoch + 1, "val_f1": val["macro_f1"], "seconds": time.time() - start})
        study.log(f"lora r{rank} b{blocks} {condition} s{seed} epoch {epoch + 1}: val={val['macro_f1']:.4f}")
        if val["macro_f1"] > best_val:
            best_val, best_params, best_epoch, stale = (
                val["macro_f1"],
                [p.detach().clone() for p in params],
                epoch + 1,
                0,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    for p, kept in zip(params, best_params, strict=True):
        p.data.copy_(kept)
    train_seconds = time.time() - start

    test, predictions = study.evaluate(
        backbone,
        head,
        study.image_loader(ctx.cache, ctx.splits["future_test"], EVAL_BATCH, WORKERS, alpha),
        ctx.classes,
    )
    extra = {}
    for name, (strength, region) in ctx.test_conditions().items():
        loader = study.image_loader(
            ctx.cache, ctx.splits["future_test"], EVAL_BATCH, WORKERS, strength, cloud_region=region
        )
        metrics, preds = study.evaluate(backbone, head, loader, ctx.classes)
        extra[name] = {"future_test": metrics, "predictions": preds.tolist()}
    result = dict(
        rank=rank,
        blocks=blocks,
        lr=lr,
        condition=condition,
        seed=seed,
        best_epoch=best_epoch,
        best_val=best_val,
        history=history,
        train_seconds=train_seconds,
        trainable_parameters=int(sum(p.numel() for p in params)),
        future_test=test,
        predictions=predictions.tolist(),
        extra=extra,
    )
    del optimizer
    return result, (backbone, head)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=6, help="number of final backbone blocks that carry LoRA")
    parser.add_argument(
        "--condition", choices=("clean", "cloud"), default="clean", help="train on clean images or with the overlay"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="cap every split (smoke test only)")
    parser.add_argument("--export", type=Path, help="also save the trained adapters here (safetensors)")
    args = parser.parse_args(argv)
    ctx = load_context(args.limit)
    out = ctx.runs / "lora_fair"
    out.mkdir(exist_ok=True)
    path = out / output_name(args.condition, args.rank, args.lr, args.blocks, args.seed)
    if path.exists() and not (args.limit or args.export):
        study.log("exists, skipping", path.name)
        return
    result, (backbone, head) = train_and_score(
        ctx, args.rank, args.blocks, args.condition, args.seed, args.lr, args.max_epochs, args.patience
    )
    if args.export:
        names = sorted({r["category"] for r in ctx.rows})
        tensors = {**adapter_state(backbone), "historical.weight": head.weight, "historical.bias": head.bias}
        save_model(
            args.export,
            "lora",
            tensors,
            names,
            rank=args.rank,
            blocks=args.blocks,
            seed=args.seed,
            val_f1=result["best_val"],
        )
    if not args.limit and not path.exists():
        study.write_json(path, result)
    study.log("done", path.name, {k: round(v["future_test"]["macro_f1"], 4) for k, v in result["extra"].items()})


if __name__ == "__main__":
    main()
