"""DINOv2-L FMoW study: harm gate, cloud calibration and matched anchoring arms.

Contract
- Splits are fixed by location group (country:UTM): historical 2010-13 train/val,
  adapt 2014-15 train/val, and the future 2016-17 test half used by study B.
- Every choice (head checkpoint, cloud alpha, anchor lambda) uses historical-val or
  adapt-val. Future-test labels are read only to score finished models.
- Arms differ only in the patch weights of the feature anchor:
  plain (no anchor), uniform, shuffled (saliency-sized random patches) and saliency
  (top-40% true-class evidence of the frozen model).

Run as `python -m fyp_drift.evaluation.fmow_dinov2l_study --output DIR`. Each stage
writes its result file and is skipped when that file exists, so a run can resume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
PERIODS = ("historical_2010_13", "adapt_2014_15", "future_2016_17")
ANCHOR_MODES = ("uniform", "shuffled", "saliency")
SALIENT_FRACTION = 0.4
DEFAULT_MANIFEST = Path("/mnt/sdd/FYP/data/fmow_v1.1/subset_7k_country_matched/manifest_downloaded.csv")
DEFAULT_CACHE = Path("/mnt/sdd/FYP/cache/fmow21k_u8_224.npy")
DEFAULT_METADATA = Path("/mnt/sdd/FYP/data/fmow_v1.1/subset_7k_country_matched/acquisition_metadata.csv")


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def write_json(path, obj):
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(obj, indent=2, default=float))
    temporary.replace(path)


# --------------------------------------------------------------------------- data
def load_rows(manifest, metadata=DEFAULT_METADATA):
    """Downloaded manifest rows with a dense class id, row index, location group and, when present, acquisition metadata."""
    with open(manifest, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("local_path") and Path(r["local_path"]).exists()]
    if Path(metadata).exists():
        with open(metadata, newline="", encoding="utf-8") as handle:
            acquisition = {m["key"]: m for m in csv.DictReader(handle)}
        for row in rows:
            extra = acquisition.get(row["key"], {})
            row.update({k: float(v) for k, v in extra.items() if k != "key" and v not in ("", None)})
    classes = sorted({r["category"] for r in rows})
    class_id = {name: i for i, name in enumerate(classes)}
    for i, row in enumerate(rows):
        row.update(class_id=class_id[row["category"]], idx=i, group=f"{row['country']}:{row['utm']}")
    return rows, len(classes)


def split_by_group(rows, held_out_fraction, seed):
    groups = [r["group"] for r in rows]
    kept, held = next(GroupShuffleSplit(n_splits=1, test_size=held_out_fraction, random_state=seed).split(rows, groups=groups))
    return [rows[i] for i in kept], [rows[i] for i in held]


def study_splits(rows):
    by_period = {p: sorted((r for r in rows if r["period"] == p), key=lambda r: r["local_path"]) for p in PERIODS}
    hist_train, hist_val = split_by_group(by_period[PERIODS[0]], 0.15, 42)
    adapt_train, adapt_val = split_by_group(by_period[PERIODS[1]], 0.1, 42)
    _, future_test = split_by_group(by_period[PERIODS[2]], 0.5, 43)
    return {"hist_train": hist_train, "hist_val": hist_val, "adapt_train": adapt_train,
            "adapt_val": adapt_val, "future_test": future_test, "future_all": by_period[PERIODS[2]]}


def image_cache(rows, path, workers):
    """uint8 224x224 RGB array indexed by row idx; decoded once, then memory-mapped."""
    path = Path(path)
    finished = path.with_suffix(".done")
    if finished.exists():
        return np.load(path, mmap_mode="r")
    cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(rows), 224, 224, 3))

    class Decode(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            with Image.open(rows[i]["local_path"]) as image:
                image.draft("RGB", (448, 448))
                pixels = np.asarray(image.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.uint8)
            return i, torch.from_numpy(pixels.copy())

    for ids, images in DataLoader(Decode(), batch_size=64, num_workers=workers):
        cache[ids.numpy()] = images.numpy()
    cache.flush()
    finished.write_text("ok")
    return np.load(path, mmap_mode="r")


CLOUD_REGIONS = ("off_object", "on_object")


def cloud_mask(key, seed=42, size=224):
    """Main-run cloud: the top 20% of a smoothed per-image noise field."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    noise = torch.rand((1, 1, 16, 16), generator=torch.Generator().manual_seed(int.from_bytes(digest[:8], "little")))
    smooth = F.avg_pool2d(noise, 5, stride=1, padding=2)
    mask = (smooth >= torch.quantile(smooth, 0.8)).float()
    return F.interpolate(mask, size=(size, size), mode="bilinear", align_corners=False)[0]


def add_cloud(image, key, alpha, seed=42):
    """Blend the main-run near-white cloud into a normalized image; same key, same cloud."""
    mask = cloud_mask(key, seed, image.shape[-1])
    white = (1 - MEAN) / STD
    return image * (1 - alpha * mask) + white * (alpha * mask)


def fractal_field(key, size=224, octaves=5, seed=42):
    """Multi-octave value noise in [0, 1], the usual cumulus-like texture for simulated clouds."""
    generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f"{seed}:{key}:fbm".encode()).digest()[:8], "little"))
    field = torch.zeros(1, 1, size, size)
    for octave in range(octaves):
        grid = torch.rand((1, 1, 4 * 2 ** octave, 4 * 2 ** octave), generator=generator)
        field += 0.5 ** octave * F.interpolate(grid, size=(size, size), mode="bicubic", align_corners=False)
    field = field[0, 0]
    return (field - field.min()) / (field.max() - field.min())


def object_box_mask(row, size=224, margin=0.1):
    """1 inside the image's labelled object box (plus a margin), 0 elsewhere."""
    mask = torch.zeros(size, size)
    half_w, half_h = (row["box_w"] / 2 + margin) * size, (row["box_h"] / 2 + margin) * size
    cx, cy = row["box_cx"] * size, row["box_cy"] * size
    mask[max(0, int(cy - half_h)):min(size, int(cy + half_h) + 1), max(0, int(cx - half_w)):min(size, int(cx + half_w) + 1)] = 1
    return mask


def cumulus_opacity(row, region, size=224, clouds=6, seed=42):
    """Opacity of a few puffy cumulus clouds: soft round envelopes carrying fractal texture.

    Cloud centres are drawn outside the labelled object box for off_object and inside it
    for on_object, so clouds keep natural rounded edges instead of being clipped to the box.
    """
    generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f"{seed}:{row['key']}:cumulus".encode()).digest()[:8], "little"))
    box = object_box_mask(row, size).bool()
    candidates = (~box if region == "off_object" else box).nonzero()
    if len(candidates) == 0:
        return torch.zeros(size, size)
    ys, xs = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    envelope = torch.zeros(size, size)
    for _ in range(clouds):
        cy, cx = candidates[torch.randint(len(candidates), (1,), generator=generator)].float()[0]
        radius = 8 + 14 * torch.rand(1, generator=generator).item()
        envelope = torch.maximum(envelope, torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * radius ** 2)))
    texture = fractal_field(row["key"], size, seed=seed)
    return ((envelope * (0.6 + 0.8 * texture) - 0.45) / 0.35).clamp(0, 1)


def realistic_cloud(pixels, row, strength, region, seed=42):
    """Composite cumulus clouds with thin edges and a sun-consistent shadow onto [0, 1] pixels.

    The shadow is the cloud opacity displaced away from the recorded sun azimuth by an
    offset that grows as the recorded sun elevation falls, and it darkens the ground beneath.
    """
    opacity = cumulus_opacity(row, region, pixels.shape[-1], seed=seed) * strength
    elevation = math.radians(max(5.0, float(row["sun_elevation"])))
    offset = min(40.0, 14.0 / math.tan(elevation))
    azimuth = math.radians(float(row["sun_azimuth"]))
    dx, dy = int(round(-offset * math.sin(azimuth))), int(round(offset * math.cos(azimuth)))
    shadow = torch.roll(opacity, shifts=(dy, dx), dims=(0, 1)) * (1 - opacity)
    shaded = pixels * (1 - 0.6 * shadow)
    brightness = 0.88 + 0.12 * fractal_field(row["key"] + ":shade", pixels.shape[-1], seed=seed)
    cloud_colour = torch.stack([0.96 * brightness, 0.97 * brightness, brightness])
    return shaded * (1 - opacity) + cloud_colour * opacity


class CachedImages(Dataset):
    """Cached images, optionally with the main-run cloud (region None) or a realistic regional cloud."""

    def __init__(self, cache, rows, cloud_alpha, cloud_region=None):
        self.cache, self.rows, self.cloud_alpha, self.cloud_region = cache, rows, cloud_alpha, cloud_region

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        pixels = torch.from_numpy(np.array(self.cache[row["idx"]])).permute(2, 0, 1).float() / 255.0
        if self.cloud_alpha > 0 and self.cloud_region:
            pixels = realistic_cloud(pixels, row, self.cloud_alpha, self.cloud_region)
        image = (pixels - MEAN) / STD
        if self.cloud_alpha > 0 and not self.cloud_region:
            image = add_cloud(image, row["local_path"], self.cloud_alpha)
        return image, row["class_id"], i


def image_loader(cache, rows, batch_size, workers, cloud_alpha=0.0, shuffle_seed=None, cloud_region=None):
    """Evaluation order when shuffle_seed is None; otherwise a seeded shuffle that drops the last batch."""
    shuffle = shuffle_seed is not None
    generator = torch.Generator().manual_seed(shuffle_seed) if shuffle else None
    return DataLoader(CachedImages(cache, rows, cloud_alpha, cloud_region), batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                      num_workers=workers, pin_memory=True, generator=generator)


# --------------------------------------------------------------------------- model
class LoRALinear(nn.Module):
    """Frozen linear layer plus a zero-initialised low-rank update that can be switched off."""

    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.enabled = True

    def forward(self, x):
        output = self.base(x)
        return output + 2.0 * self.up(self.down(x)) if self.enabled else output


class Backbone(nn.Module):
    """DINOv2 whose first blocks run without gradients and whose last blocks carry LoRA.

    With adapters disabled the model is exactly the frozen backbone, so the anchor
    reference reuses the same weights and the same early-block activations.
    """

    def __init__(self, name, trainable_blocks=0, rank=8):
        super().__init__()
        from transformers import Dinov2Model
        self.model = Dinov2Model.from_pretrained(name)
        self.model.requires_grad_(False)
        self.width = self.model.config.hidden_size
        blocks = list(self.model.encoder.layer)
        self.frozen_blocks, self.adapted_blocks = blocks[:len(blocks) - trainable_blocks], blocks[len(blocks) - trainable_blocks:]
        self.adapters = []
        for block in self.adapted_blocks:
            attention = block.attention.attention
            sites = [(attention, "query"), (attention, "key"), (attention, "value"),
                     (block.attention.output, "dense"), (block.mlp, "fc1"), (block.mlp, "fc2")]
            for parent, name_ in sites:
                adapter = LoRALinear(getattr(parent, name_), rank)
                setattr(parent, name_, adapter)
                self.adapters.append(adapter)

    def use_adapters(self, enabled):
        for adapter in self.adapters:
            adapter.enabled = enabled

    def early_features(self, images):
        with torch.no_grad():
            hidden = self.model.embeddings(images)
            for block in self.frozen_blocks:
                hidden = _block_output(block, hidden)
        return hidden

    def late_features(self, hidden):
        for block in self.adapted_blocks:
            hidden = _block_output(block, hidden)
        return self.model.layernorm(hidden)

    def forward(self, images):
        return self.late_features(self.early_features(images))


def _block_output(block, hidden):
    output = block(hidden)
    return output[0] if isinstance(output, (tuple, list)) else output


def patch_mean(tokens):
    return tokens[:, 1:].float().mean(1)


def scores(labels, predictions, classes):
    counts = np.bincount(predictions, minlength=classes)
    return {"samples": int(len(labels)), "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=np.arange(classes), average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "unique_predictions": int((counts > 0).sum()), "max_class_share": float(counts.max() / max(1, len(predictions)))}


@torch.no_grad()
def evaluate(backbone, head, data, classes):
    backbone.eval()
    labels, predictions = [], []
    for images, targets, _ in data:
        with torch.autocast("cuda", dtype=torch.float16):
            tokens = backbone(images.cuda(non_blocking=True))
        predictions.append(head(patch_mean(tokens)).argmax(1).cpu())
        labels.append(targets)
    labels, predictions = torch.cat(labels).numpy(), torch.cat(predictions).numpy()
    return scores(labels, predictions, classes), predictions


def cluster_bootstrap(values, groups, repetitions=2000, seed=0):
    """Mean of per-image values with a 95% interval from resampling whole location groups."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    by_group = {g: values[groups == g] for g in unique}
    draws = [np.concatenate([by_group[g] for g in rng.choice(unique, len(unique))]).mean() for _ in range(repetitions)]
    return {"mean": float(values.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


# --------------------------------------------------------------------------- stages
def frozen_features(args, cache, rows):
    path = args.output / "features.npz"
    if not path.exists():
        backbone = Backbone(args.backbone).cuda().eval()
        means = []
        with torch.no_grad():
            for images, _, _ in image_loader(cache, rows, args.eval_batch, args.workers):
                with torch.autocast("cuda", dtype=torch.float16):
                    means.append(patch_mean(backbone(images.cuda(non_blocking=True))).half().cpu())
        np.savez(path, mean=torch.cat(means).numpy())
        del backbone
        torch.cuda.empty_cache()
    return np.load(path)["mean"].astype(np.float32)


def fit_linear_head(features, labels, classes, val_features, val_labels, epochs=300):
    """Class-weighted linear head; keeps the checkpoint with the best validation macro-F1."""
    torch.manual_seed(42)
    x, y = torch.tensor(features).cuda(), torch.tensor(labels).cuda()
    xv = torch.tensor(val_features).cuda()
    counts = torch.bincount(y, minlength=classes).float().clamp_min(1)
    head = nn.Linear(x.shape[1], classes).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1, best_state = -1.0, None
    for epoch in range(epochs):
        for batch in torch.randperm(len(x), device="cuda").split(512):
            loss = F.cross_entropy(head(x[batch]), y[batch], weight=counts.sum() / (classes * counts))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 != 9:
            continue
        with torch.no_grad():
            val_f1 = f1_score(val_labels, head(xv).argmax(1).cpu().numpy(), average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head.eval(), best_f1


def drift_statistics(features, index, permutations):
    """MMD and domain AUC on 1,000 frozen features per group, with a historical split-half null."""
    from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
    rng = np.random.default_rng(42)
    historical = features[rng.choice(np.concatenate([index["hist_train"], index["hist_val"]]), 2000, replace=False)]
    future = features[rng.choice(index["future_all"], 1000, replace=False)]
    adapt = features[rng.choice(np.concatenate([index["adapt_train"], index["adapt_val"]]), 1000, replace=False)]
    pairs = {"historical_vs_future": (historical[:1000], future), "historical_vs_adapt": (historical[:1000], adapt),
             "historical_split_half_null": (historical[:1000], historical[1000:])}
    results = []
    for name, (first, second) in pairs.items():
        metrics = distribution_shift_metrics(first, second, permutations=permutations, permutation_limit=1000, seed=42)
        results.append({"comparison": name, **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}})
    return results


def run_harm_gate(args, features, splits, classes):
    path = args.output / "gate.json"
    if path.exists():
        return json.loads(path.read_text()), torch.load(args.output / "head.pt", weights_only=True)
    index = {k: np.array([r["idx"] for r in v]) for k, v in splits.items()}
    labels = {k: np.array([r["class_id"] for r in v]) for k, v in splits.items()}
    head, val_f1 = fit_linear_head(features[index["hist_train"]], labels["hist_train"], classes,
                                   features[index["hist_val"]], labels["hist_val"])
    torch.save(head.state_dict(), args.output / "head.pt")
    with torch.no_grad():
        predicted = {k: head(torch.tensor(features[index[k]]).cuda()).argmax(1).cpu().numpy() for k in index}
    window_scores = {k: scores(labels[k], predicted[k], classes) for k in ("hist_val", "adapt_val", "adapt_train", "future_test", "future_all")}
    gap = accuracy_gap(splits, predicted, labels)
    gate = {"backbone": args.backbone, "head_hist_val_macro_f1": val_f1, "scores": window_scores,
            "drift": drift_statistics(features, index, args.permutations),
            "accuracy_gap_hist_val_minus_future": gap, "counts": {k: len(v) for k, v in splits.items()},
            "harm_gate_passed": bool(gap["mean"] >= 0.03 and gap["ci95"][0] > 0)}
    write_json(path, gate)
    return gate, head.state_dict()


def accuracy_gap(splits, predicted, labels, repetitions=1000):
    """Historical-val minus future-test accuracy, resampling location groups in each set independently."""
    rng = np.random.default_rng(0)
    sides = []
    for name in ("hist_val", "future_test"):
        groups = np.array([r["group"] for r in splits[name]])
        correct = (predicted[name] == labels[name]).astype(float)
        sides.append((np.unique(groups), {g: correct[groups == g] for g in np.unique(groups)}))
    draws = []
    for _ in range(repetitions):
        means = [np.concatenate([by_group[g] for g in rng.choice(unique, len(unique))]).mean() for unique, by_group in sides]
        draws.append(means[0] - means[1])
    return {"mean": float(np.mean(draws)), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def calibrate_cloud(args, cache, splits, head_state, classes):
    """Pick the cloud alpha whose historical-val macro-F1 drop is closest to 7.5 points."""
    path = args.output / "calib.json"
    if path.exists():
        return json.loads(path.read_text())
    backbone = Backbone(args.backbone).cuda().eval()
    head = nn.Linear(backbone.width, classes).cuda()
    head.load_state_dict(head_state)
    candidates = []
    for alpha in (0.0, 0.2, 0.35, 0.5, 0.7, 0.85):
        metrics, _ = evaluate(backbone, head, image_loader(cache, splits["hist_val"], args.eval_batch, args.workers, alpha), classes)
        candidates.append({"alpha": alpha, **metrics})
    clean_f1 = candidates[0]["macro_f1"]
    chosen = min(candidates[1:], key=lambda c: abs((clean_f1 - c["macro_f1"]) - 0.075))
    drop = clean_f1 - chosen["macro_f1"]
    result = {"candidates": candidates, "selected_alpha": chosen["alpha"], "selected_drop": drop,
              "within_5_10_points": bool(0.05 <= drop <= 0.10)}
    write_json(path, result)
    del backbone
    torch.cuda.empty_cache()
    return result


def anchor_weights(reference_tokens, head, labels, mode, generator):
    """Per-patch anchor weights summing to one for each image."""
    patch_count = reference_tokens.shape[1]
    if mode == "uniform":
        return torch.full(reference_tokens.shape[:2], 1.0 / patch_count, device=reference_tokens.device)
    evidence = (reference_tokens * head.weight[labels][:, None, :]).sum(-1)
    salient = round(SALIENT_FRACTION * patch_count)
    weights = torch.zeros_like(evidence).scatter_(1, evidence.topk(salient, dim=1).indices, 1.0 / salient)
    if mode == "shuffled":
        weights = weights.gather(1, torch.rand(weights.shape, generator=generator).argsort(1).to(weights.device))
    return weights


def train_arm(args, cache, splits, head, classes, mode, anchor_lambda, cloud_alpha, seed, extra_tests=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone = Backbone(args.backbone, args.trainable_blocks, args.rank).cuda()
    parameters = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(seed + 7)
    accumulation = max(1, args.effective_batch // args.batch)
    step0 = step_zero_gap(backbone, head, cache, splits, cloud_alpha)
    history = []
    for epoch in range(args.epochs):
        backbone.train()
        losses = {"ce": 0.0, "anchor": 0.0}
        batches = image_loader(cache, splits["adapt_train"], args.batch, args.workers, cloud_alpha, shuffle_seed=seed + epoch)
        optimizer.zero_grad(set_to_none=True)
        for step, (images, targets, _) in enumerate(batches):
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                early = backbone.early_features(images)
                tokens = backbone.late_features(early)[:, 1:].float()
                ce = F.cross_entropy(head(tokens.mean(1)), targets)
                anchor = anchor_penalty(backbone, head, early, tokens, targets, mode, generator) if mode != "plain" else torch.zeros((), device="cuda")
                loss = (ce + anchor_lambda * anchor) / accumulation
            scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses["ce"] += float(ce)
            losses["anchor"] += float(anchor)
        val, _ = evaluate(backbone, head, image_loader(cache, splits["adapt_val"], args.eval_batch, args.workers, cloud_alpha), classes)
        history.append({"epoch": epoch + 1, "ce": losses["ce"] / (step + 1), "anchor_mse": losses["anchor"] / (step + 1), "adapt_val": val})
        log(f"  {mode} lambda={anchor_lambda} alpha={cloud_alpha} seed={seed} epoch={epoch + 1} val_f1={val['macro_f1']:.4f}")
    test, predictions = evaluate(backbone, head, image_loader(cache, splits["future_test"], args.eval_batch, args.workers, cloud_alpha), classes)
    shifted = {}
    for name, (alpha, region) in (extra_tests or {}).items():
        metrics, shifted_predictions = evaluate(backbone, head, image_loader(cache, splits["future_test"], args.eval_batch, args.workers, alpha, cloud_region=region), classes)
        shifted[name] = {"future_test": metrics, "predictions": shifted_predictions.tolist()}
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    del backbone, optimizer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {"mode": mode, "lambda": anchor_lambda, "cloud_alpha": cloud_alpha, "seed": seed, "step0_max_logit_diff": step0,
            "trainable_parameters": int(sum(p.numel() for p in parameters)), "history": history, "adapt_val": history[-1]["adapt_val"],
            "future_test": test, "predictions": predictions.tolist(), "shifted_tests": shifted, "peak_gpu_gib": peak_gib}


def anchor_penalty(backbone, head, early, tokens, targets, mode, generator):
    """Weighted squared distance between adapted and frozen patch tokens."""
    with torch.no_grad():
        backbone.use_adapters(False)
        reference = backbone.late_features(early)[:, 1:].float()
        backbone.use_adapters(True)
        weights = anchor_weights(reference, head, targets, mode, generator)
    return (weights * (tokens - reference).square().mean(-1)).sum(1).mean()


def step_zero_gap(backbone, head, cache, splits, cloud_alpha):
    """Largest logit difference between adapters off and freshly initialised adapters on (should be 0)."""
    images = next(iter(image_loader(cache, splits["adapt_val"][:8], 8, 0, cloud_alpha)))[0].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        backbone.use_adapters(False)
        frozen = head(patch_mean(backbone(images)))
        backbone.use_adapters(True)
        adapted = head(patch_mean(backbone(images)))
    return float((frozen - adapted).abs().max())


def run_arms(args, cache, splits, head_state, classes, cloud_alpha):
    head =nn.Linear(head_state["weight"].shape[1], classes).cuda().eval()
    head.load_state_dict(head_state)
    head.requires_grad_(False)
    arms = args.output / "arms"
    arms.mkdir(exist_ok=True)
    conditions = {"clean": 0.0, "cloud": cloud_alpha}
    score_frozen_model(args, cache, splits, head, classes, conditions, arms)
    for seed in args.seeds:
        for condition, alpha in conditions.items():
            for mode, anchor_lambda in [("plain", 0.0)] + [(m, l) for m in ANCHOR_MODES for l in args.lambdas]:
                path = arms / f"{condition}_{mode}_l{anchor_lambda}_s{seed}.json"
                if path.exists():
                    continue
                started = time.time()
                result = train_arm(args, cache, splits, head, classes, mode, anchor_lambda, alpha, seed)
                write_json(path, {**result, "condition": condition, "seconds": time.time() - started})
                log(f"done {path.name}: test_f1={result['future_test']['macro_f1']:.4f} peak={result['peak_gpu_gib']:.1f}GiB")


def score_frozen_model(args, cache, splits, head, classes, conditions, arms):
    backbone = Backbone(args.backbone).cuda().eval()
    for condition, alpha in conditions.items():
        path = arms / f"frozen_{condition}.json"
        if path.exists():
            continue
        metrics, predictions = evaluate(backbone, head, image_loader(cache, splits["future_test"], args.eval_batch, args.workers, alpha), classes)
        write_json(path, {"mode": "frozen", "condition": condition, "future_test": metrics, "predictions": predictions.tolist()})
    del backbone
    torch.cuda.empty_cache()


def summarize(args, splits):
    """Lambda per (condition, mode) chosen on mean adapt-val F1; paired deltas on future-test."""
    results = [json.loads(p.read_text()) for p in sorted((args.output / "arms").glob("*_s*.json"))]
    labels = np.array([r["class_id"] for r in splits["future_test"]])
    groups = np.array([r["group"] for r in splits["future_test"]])
    chosen = {}
    for condition in ("clean", "cloud"):
        for mode in ANCHOR_MODES:
            by_lambda = {}
            for r in results:
                if r["condition"] == condition and r["mode"] == mode:
                    by_lambda.setdefault(r["lambda"], []).append(r["adapt_val"]["macro_f1"])
            if by_lambda:
                chosen[(condition, mode)] = max(by_lambda, key=lambda l: np.mean(by_lambda[l]))

    def selected(condition, mode, seed):
        anchor_lambda = 0.0 if mode == "plain" else chosen.get((condition, mode))
        return next((r for r in results if (r["condition"], r["mode"], r["seed"], r["lambda"]) == (condition, mode, seed, anchor_lambda)), None)

    means, paired = {}, []
    for condition in ("clean", "cloud"):
        for mode in ("plain",) + ANCHOR_MODES:
            runs = [selected(condition, mode, s) for s in args.seeds if selected(condition, mode, s)]
            if runs:
                f1s = [r["future_test"]["macro_f1"] for r in runs]
                means[f"{condition}/{mode}"] = {"lambda": runs[0]["lambda"], "seeds": len(f1s), "mean_test_macro_f1": float(np.mean(f1s)),
                                                "sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None}
        for a, b in (("saliency", "uniform"), ("saliency", "shuffled"), ("saliency", "plain"), ("uniform", "plain")):
            for seed in args.seeds:
                first, second = selected(condition, a, seed), selected(condition, b, seed)
                if first and second:
                    accuracy_difference = (np.array(first["predictions"]) == labels).astype(float) - (np.array(second["predictions"]) == labels).astype(float)
                    paired.append({"condition": condition, "a": a, "b": b, "seed": seed,
                                   "delta_macro_f1": first["future_test"]["macro_f1"] - second["future_test"]["macro_f1"],
                                   "delta_accuracy": cluster_bootstrap(accuracy_difference, groups)})
    table = [{"condition": r["condition"], "mode": r["mode"], "lambda": r["lambda"], "seed": r["seed"],
              "adapt_val_f1": r["adapt_val"]["macro_f1"], **{f"test_{k}": r["future_test"][k] for k in ("macro_f1", "accuracy", "unique_predictions")}}
             for r in results]
    write_json(args.output / "summary.json", {"chosen_lambda": {f"{c}/{m}": l for (c, m), l in chosen.items()},
                                              "means": means, "paired": paired, "all_runs": table})
    return means


def calibrate_region_clouds(args, cache, splits, head, classes):
    """Per region, the cloud strength whose frozen historical-val macro-F1 drop is closest to 7.5 points."""
    path = args.output / "calib_regions.json"
    if path.exists():
        return json.loads(path.read_text())
    missing = [r["key"] for r in splits["hist_val"] + splits["future_test"] if "sun_elevation" not in r or "box_cx" not in r]
    if missing:
        raise RuntimeError(f"{len(missing)} images lack acquisition metadata; run fyp_drift.data.fmow_metadata first")
    backbone = Backbone(args.backbone).cuda().eval()
    result = {}
    for region in ("off_object", "on_object"):
        candidates = []
        for alpha in (0.0, 0.4, 0.6, 0.8, 1.0):
            metrics, _ = evaluate(backbone, head, image_loader(cache, splits["hist_val"], args.eval_batch, args.workers, alpha, cloud_region=region), classes)
            candidates.append({"alpha": alpha, **metrics})
        clean_f1 = candidates[0]["macro_f1"]
        chosen = min(candidates[1:], key=lambda c: abs((clean_f1 - c["macro_f1"]) - 0.075))
        result[region] = {"candidates": candidates, "selected_alpha": chosen["alpha"], "selected_drop": clean_f1 - chosen["macro_f1"]}
        log("region calib", region, result[region]["selected_alpha"], round(result[region]["selected_drop"], 4))
    write_json(path, result)
    del backbone
    torch.cuda.empty_cache()
    return result


def run_unseen_cloud_arms(args, cache, splits, head_state, classes):
    """Adapt on clean images with each arm's adapt-val lambda, then score on unseen realistic clouds off and on the object."""
    head = nn.Linear(head_state["weight"].shape[1], classes).cuda().eval()
    head.load_state_dict(head_state)
    head.requires_grad_(False)
    chosen = json.loads((args.output / "summary.json").read_text())["chosen_lambda"]
    regions = calibrate_region_clouds(args, cache, splits, head, classes)
    tests = {region: (regions[region]["selected_alpha"], region) for region in regions}
    out = args.output / "unseen_cloud"
    out.mkdir(exist_ok=True)
    backbone = Backbone(args.backbone).cuda().eval()
    for name, (alpha, region) in tests.items():
        path = out / f"frozen_{name}.json"
        if not path.exists():
            metrics, predictions = evaluate(backbone, head, image_loader(cache, splits["future_test"], args.eval_batch, args.workers, alpha, cloud_region=region), classes)
            write_json(path, {"mode": "frozen", "condition": name, "future_test": metrics, "predictions": predictions.tolist()})
    del backbone
    torch.cuda.empty_cache()
    for seed in args.seeds:
        for mode in ("plain",) + ANCHOR_MODES:
            anchor_lambda = 0.0 if mode == "plain" else float(chosen[f"clean/{mode}"])
            path = out / f"{mode}_l{anchor_lambda}_s{seed}.json"
            if path.exists():
                continue
            result = train_arm(args, cache, splits, head, classes, mode, anchor_lambda, 0.0, seed, extra_tests=tests)
            write_json(path, {**result, "condition": "clean_adapt_unseen_cloud"})
            log(f"done {path.name}: clean={result['future_test']['macro_f1']:.4f} " + " ".join(f"{k}={v['future_test']['macro_f1']:.4f}" for k, v in result["shifted_tests"].items()))
    summarize_unseen(args, splits)


def summarize_unseen(args, splits):
    labels = np.array([r["class_id"] for r in splits["future_test"]])
    groups = np.array([r["group"] for r in splits["future_test"]])
    runs = [json.loads(p.read_text()) for p in sorted((args.output / "unseen_cloud").glob("*_s*.json"))]
    frozen = {p.stem.replace("frozen_", ""): json.loads(p.read_text())["future_test"] for p in (args.output / "unseen_cloud").glob("frozen_*.json")}
    means, paired = {}, []
    tests = ("clean",) + tuple(runs[0]["shifted_tests"]) if runs else ()
    for test in tests:
        def f1(run):
            return run["future_test"]["macro_f1"] if test == "clean" else run["shifted_tests"][test]["future_test"]["macro_f1"]

        def predictions(run):
            return np.array(run["predictions"] if test == "clean" else run["shifted_tests"][test]["predictions"])
        for mode in ("plain",) + ANCHOR_MODES:
            values = [f1(r) for r in runs if r["mode"] == mode]
            if values:
                means[f"{test}/{mode}"] = {"seeds": len(values), "mean_test_macro_f1": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        for a, b in (("saliency", "uniform"), ("saliency", "shuffled"), ("saliency", "plain")):
            for seed in args.seeds:
                first = next((r for r in runs if r["mode"] == a and r["seed"] == seed), None)
                second = next((r for r in runs if r["mode"] == b and r["seed"] == seed), None)
                if first and second:
                    difference = (predictions(first) == labels).astype(float) - (predictions(second) == labels).astype(float)
                    paired.append({"test": test, "a": a, "b": b, "seed": seed, "delta_macro_f1": f1(first) - f1(second),
                                   "delta_accuracy": cluster_bootstrap(difference, groups)})
    write_json(args.output / "summary_unseen_cloud.json", {"frozen": frozen, "means": means, "paired": paired})
    log("unseen-cloud summary", json.dumps(means, indent=1))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="facebook/dinov2-large")
    parser.add_argument("--stages", nargs="+", choices=("gate", "calib", "arms", "summary", "unseen_cloud"), default=["gate", "calib", "arms", "summary"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--trainable-blocks", type=int, default=6)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--limit", type=int, default=0, help="cap every split for an execution check")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", {k: str(v) for k, v in vars(args).items()})
    rows, classes = load_rows(args.manifest)
    splits = study_splits(rows)
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}
    cache = image_cache(rows, args.cache, args.workers)
    gate, head_state = run_harm_gate(args, frozen_features(args, cache, rows), splits, classes)
    log("gate", gate["harm_gate_passed"], gate["accuracy_gap_hist_val_minus_future"])
    if "calib" in args.stages or "arms" in args.stages:
        cloud = calibrate_cloud(args, cache, splits, head_state, classes)
    if "arms" in args.stages:
        run_arms(args, cache, splits, head_state, classes, cloud["selected_alpha"])
    if "summary" in args.stages:
        log("summary", json.dumps(summarize(args, splits), indent=1))
    if "unseen_cloud" in args.stages:
        run_unseen_cloud_arms(args, cache, splits, head_state, classes)


if __name__ == "__main__":
    main()
