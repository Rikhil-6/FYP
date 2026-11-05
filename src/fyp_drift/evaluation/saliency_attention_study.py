"""Saliency-essential attention adapters on a frozen DINOv2-L, with faithfulness and grounding tests.

Frozen final patch tokens are reused from fyp_drift.evaluation.saliency_adapters. Every arm
is a small attention-pooling head over frozen tokens, initialised from the historical
linear head, adapted on 2014-15 images, checkpointed on adapt-val, scored on the 2016-17
test half. Saliency z is the frozen model's per-patch evidence for its own predicted class,
standardised per image, so no labels are needed at test time.

Arms (each saliency arm has a *_shuffled twin that receives z permuted within the image):
  attention  learned logits s(x)                        w = softmax(s)
  guided     attention + KL(softmax(z) || w)            saliency as a soft target
  prior      w = softmax(beta * z + s), s starts at 0    saliency is the starting attention
  gated      w ~ softmax(s) * sigmoid(gamma * z + delta) saliency gates what can be read
  input      w = softmax(s([token, z]))                 saliency is an input feature

Tests on the test half, per arm and seed:
  faithfulness  accuracy and prediction flips when the learned weights are replaced by
                uniform or within-image shuffled weights, and when the top / bottom / random
                10, 25, 50% of weighted patches are deleted
  grounding     attention mass inside the labelled fMoW object box versus the box's patch
                share, pointing-game hit rate, and (cloud condition) mass on cloud patches
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
    DEFAULT_MANIFEST, cloud_mask, cluster_bootstrap, load_rows, log, scores, study_splits, write_json,
)
from fyp_drift.evaluation.saliency_adapters import SPLITS, batches

SALIENCY_ARMS = ("guided", "prior", "gated", "input")
ARMS = ("attention",) + SALIENCY_ARMS + tuple(f"{a}_shuffled" for a in SALIENCY_ARMS)
from fyp_drift.evaluation.fmow_dinov2l_study import IMAGE_SIZE
import os
GRID = int(os.environ.get("FYP_GRID", IMAGE_SIZE // 14))  # v2 pools 2x2 tokens -> 10x10 grid


class SaliencyAttentionHead(nn.Module):
    def __init__(self, arm, head_state, width, classes):
        super().__init__()
        self.kind, self.shuffled = arm.replace("_shuffled", ""), arm.endswith("_shuffled")
        self.frozen = nn.Linear(width, classes)
        self.frozen.load_state_dict(head_state)
        self.frozen.requires_grad_(False)
        self.classifier = nn.Linear(width, classes)
        self.classifier.load_state_dict(head_state)
        inputs = width + (1 if self.kind == "input" else 0)
        self.scorer = nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, 128), nn.GELU(), nn.Linear(128, 1))
        if self.kind == "prior":
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)
            self.beta = nn.Parameter(torch.tensor(1.0))
        if self.kind == "gated":
            self.gamma, self.delta = nn.Parameter(torch.tensor(2.0)), nn.Parameter(torch.tensor(0.0))

    def saliency(self, tokens, generator):
        """Standardised frozen evidence for the frozen prediction; permuted within image for *_shuffled."""
        with torch.no_grad():
            predicted = self.frozen(tokens.mean(1)).argmax(1)
            evidence = (tokens * self.frozen.weight[predicted][:, None, :]).sum(-1)
            z = (evidence - evidence.mean(1, keepdim=True)) / evidence.std(1, keepdim=True).clamp_min(1e-6)
            if self.shuffled:
                z = z.gather(1, torch.rand(z.shape, generator=generator).argsort(1).to(z.device))
        return z

    def weights(self, tokens, z):
        if self.kind == "input":
            return self.scorer(torch.cat([tokens, z[..., None]], -1)).squeeze(-1).softmax(1)
        logits = self.scorer(tokens).squeeze(-1)
        if self.kind == "prior":
            return (self.beta * z + logits).softmax(1)
        if self.kind == "gated":
            gated = logits.softmax(1) * torch.sigmoid(self.gamma * z + self.delta)
            return gated / gated.sum(1, keepdim=True).clamp_min(1e-8)
        return logits.softmax(1)

    def classify(self, tokens, weights):
        return self.classifier((weights[..., None] * tokens).sum(1))

    def forward(self, tokens, generator):
        z = self.saliency(tokens, generator)
        weights = self.weights(tokens, z)
        return self.classify(tokens, weights), weights, z


def train(arm, seed, head_state, width, classes, data, guide_weight, args):
    torch.manual_seed(seed)
    model = SaliencyAttentionHead(arm, head_state, width, classes).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    best = {"val_f1": -1.0}
    for epoch in range(args.epochs):
        model.train()
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        for tokens, targets in batches(*data["adapt_train"], args.batch, shuffle_seed=seed * 1000 + epoch):
            logits, weights, z = model(tokens, generator)
            loss = F.cross_entropy(logits, targets)
            if model.kind == "guided":
                loss = loss + guide_weight * F.kl_div(weights.clamp_min(1e-8).log(), z.softmax(1), reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = scores(data["adapt_val"][1], infer(model, data["adapt_val"][0], data["adapt_val"][1])[0], classes)["macro_f1"]
        if val > best["val_f1"]:
            best = {"val_f1": val, "epoch": epoch + 1, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model.eval(), best


@torch.no_grad()
def infer(model, tokens, labels):
    """Predictions, attention weights and saliency for every image, in order."""
    model.eval()
    generator = torch.Generator().manual_seed(7)
    predictions, weights, saliency = [], [], []
    for x, _ in batches(tokens, labels, 256):
        logits, w, z = model(x, generator)
        predictions.append(logits.argmax(1).cpu())
        weights.append(w.cpu())
        saliency.append(z.cpu())
    return torch.cat(predictions).numpy(), torch.cat(weights), torch.cat(saliency)


@torch.no_grad()
def faithfulness(model, tokens, labels, weights, predictions):
    """Accuracy and flip rate when the head reads replaced or partially deleted weights."""
    generator = torch.Generator().manual_seed(11)
    results = {}
    def replaced(name, transform):
        out = []
        for start, (x, _) in zip(range(0, len(labels), 256), batches(tokens, labels, 256)):
            out.append(model.classify(x, transform(weights[start:start + len(x)].cuda())).argmax(1).cpu())
        p = torch.cat(out).numpy()
        results[name] = {"accuracy": float((p == labels).mean()), "flip_rate": float((p != predictions).mean())}
    replaced("uniform", lambda w: torch.full_like(w, 1.0 / w.shape[1]))
    replaced("shuffled_weights", lambda w: w.gather(1, torch.rand(w.shape, generator=generator).argsort(1).to(w.device)))
    for fraction in (0.1, 0.25, 0.5):
        k = round(fraction * GRID * GRID)
        def delete(order_fn):
            def transform(w):
                keep = torch.ones_like(w).scatter_(1, order_fn(w)[:, :k], 0.0)
                kept = w * keep
                return kept / kept.sum(1, keepdim=True).clamp_min(1e-8)
            return transform
        replaced(f"delete_top_{fraction}", delete(lambda w: w.argsort(1, descending=True)))
        replaced(f"delete_bottom_{fraction}", delete(lambda w: w.argsort(1)))
        replaced(f"delete_random_{fraction}", delete(lambda w: torch.rand(w.shape, generator=generator).argsort(1).to(w.device)))
    return results


def box_patch_mask(row):
    """16x16 boolean grid of patches whose centres fall inside the labelled box (at least the centre patch)."""
    centres = (torch.arange(GRID) + 0.5) / GRID
    inside_x = (centres - row["box_cx"]).abs() <= row["box_w"] / 2
    inside_y = (centres - row["box_cy"]).abs() <= row["box_h"] / 2
    mask = inside_y[:, None] & inside_x[None, :]
    if not mask.any():
        mask[min(GRID - 1, int(row["box_cy"] * GRID)), min(GRID - 1, int(row["box_cx"] * GRID))] = True
    return mask.flatten()


def grounding(weights, rows, cloudy):
    boxes = torch.stack([box_patch_mask(r) for r in rows]).float()
    mass = (weights * boxes).sum(1)
    chance = boxes.mean(1)
    hit = boxes.gather(1, weights.argmax(1, keepdim=True)).squeeze(1)
    result = {"box_mass": float(mass.mean()), "box_share": float(chance.mean()), "box_mass_over_share": float((mass / chance).mean()),
              "pointing_game": float(hit.mean())}
    if cloudy:
        clouds = torch.stack([(F.avg_pool2d(cloud_mask(r["local_path"])[None], IMAGE_SIZE // GRID)[0, 0] > 0.5).flatten() for r in rows]).float()
        has_cloud = clouds.sum(1) > 0
        result |= {"cloud_mass": float((weights * clouds).sum(1)[has_cloud].mean()), "cloud_share": float(clouds.mean(1)[has_cloud].mean())}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path(os.environ.get("FYP_ROOT", "/mnt/sdd/FYP")) / "runs/today/main")
    parser.add_argument("--tokens", type=Path, default=Path(os.environ.get("FYP_ROOT", "/mnt/sdd/FYP")) / "cache/adapter_tokens")
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("FYP_ROOT", "/mnt/sdd/FYP")) / "runs/today/saliency_attention")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--guide-weights", default='{"clean": 0.3, "cloud": 0.1}', help="adapt-val choices from the guidance sweep")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--conditions", nargs="+", default=["clean", "cloud"], choices=["clean", "cloud"])
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    guide_weights = json.loads(args.guide_weights)
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    test_rows = splits["future_test"]
    labels = np.array([r["class_id"] for r in test_rows])
    groups = np.array([r["group"] for r in test_rows])
    head_state = torch.load(args.run / "head.pt", weights_only=True)
    width = head_state["weight"].shape[1]
    for condition in args.conditions:
        data = {s: (np.load(args.tokens / f"{condition}_{s}.npy", mmap_mode="r"), np.array([r["class_id"] for r in splits[s]])) for s in SPLITS}
        for seed in args.seeds:
            for arm in args.arms:
                path = args.output / f"{condition}_{arm}_s{seed}.json"
                if path.exists():
                    continue
                model, best = train(arm, seed, head_state, width, classes, data, guide_weights[condition], args)
                predictions, weights, saliency = infer(model, *data["future_test"])
                result = {"arm": arm, "condition": condition, "seed": seed, "best_epoch": best["epoch"], "adapt_val_macro_f1": best["val_f1"],
                          "future_test": scores(labels, predictions, classes), "predictions": predictions.tolist(),
                          "faithfulness": faithfulness(model, *data["future_test"], weights, predictions),
                          "grounding": grounding(weights, test_rows, condition == "cloud"),
                          "saliency_grounding": grounding(saliency.softmax(1), test_rows, condition == "cloud"),
                          "learned_scalars": {n: float(getattr(model, n)) for n in ("beta", "gamma", "delta") if hasattr(model, n)}}
                write_json(path, result)
                log(f"{condition} {arm} s{seed}: test={result['future_test']['macro_f1']:.4f} uniform-flip={result['faithfulness']['uniform']['flip_rate']:.3f} "
                    f"box={result['grounding']['box_mass_over_share']:.2f}x point={result['grounding']['pointing_game']:.3f}")
    summarize(args, labels, groups)


def summarize(args, labels, groups):
    runs = [json.loads(p.read_text()) for p in sorted(args.output.glob("*_s*.json"))]
    table, paired = {}, []
    for condition in ("clean", "cloud"):
        for arm in ARMS:
            rs = [r for r in runs if (r["condition"], r["arm"]) == (condition, arm)]
            if not rs:
                continue
            mean = lambda f: float(np.mean([f(r) for r in rs]))
            table[f"{condition}/{arm}"] = {
                "seeds": len(rs), "test_macro_f1": mean(lambda r: r["future_test"]["macro_f1"]), "test_accuracy": mean(lambda r: r["future_test"]["accuracy"]),
                "uniform_flip": mean(lambda r: r["faithfulness"]["uniform"]["flip_rate"]),
                "uniform_acc_drop": mean(lambda r: r["future_test"]["accuracy"] - r["faithfulness"]["uniform"]["accuracy"]),
                "delete_top25_acc_drop": mean(lambda r: r["future_test"]["accuracy"] - r["faithfulness"]["delete_top_0.25"]["accuracy"]),
                "delete_bottom25_acc_drop": mean(lambda r: r["future_test"]["accuracy"] - r["faithfulness"]["delete_bottom_0.25"]["accuracy"]),
                "delete_random25_acc_drop": mean(lambda r: r["future_test"]["accuracy"] - r["faithfulness"]["delete_random_0.25"]["accuracy"]),
                "box_mass_over_share": mean(lambda r: r["grounding"]["box_mass_over_share"]), "pointing_game": mean(lambda r: r["grounding"]["pointing_game"]),
                **({"cloud_mass": mean(lambda r: r["grounding"]["cloud_mass"]), "cloud_share": mean(lambda r: r["grounding"]["cloud_share"])} if condition == "cloud" else {})}
            for r in rs:
                base = next((b for b in runs if (b["condition"], b["arm"], b["seed"]) == (condition, "attention", r["seed"])), None)
                if base and arm != "attention":
                    difference = (np.array(r["predictions"]) == labels).astype(float) - (np.array(base["predictions"]) == labels).astype(float)
                    paired.append({"condition": condition, "arm": arm, "seed": r["seed"], "delta_macro_f1_vs_attention": r["future_test"]["macro_f1"] - base["future_test"]["macro_f1"],
                                   "delta_accuracy_vs_attention": cluster_bootstrap(difference, groups, repetitions=1000)})
    write_json(args.output / "summary.json", {"table": table, "paired_vs_attention": paired})
    log("summary", json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
