"""Chronological clean-drift audit and saliency/LoRA pilot on custom FMoW-RGB."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TVF
from tqdm import tqdm
from transformers import Dinov2Model

from fyp_drift.data.clear100 import MEAN, STD
from fyp_drift.data.corruptions import drift_transform
from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
from fyp_drift.data.temporal_sampling import matched_temporal_sample
from fyp_drift.evaluation.temporal_design import (
    crossed_corruption_grid,
    progressive_corruption_schedule,
)
from fyp_drift.models.vision_transformer import BACKBONE
from fyp_drift.training.saliency_lora_pilot import LoRALinear, teacher_saliency


PERIODS = ("historical_2010_13", "adapt_2014_15", "future_2016_17")
# FMoW includes verified JPEGs above Pillow's conservative pixel limit. Ask the
# JPEG decoder for a reduced-resolution decode before materializing RGB pixels.
Image.MAX_IMAGE_PIXELS = None


class FMoWImages(Dataset):
    def __init__(self, rows, corruption=None, seed=42):
        self.rows = rows
        self.corruption = corruption
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["local_path"]) as opened:
            opened.draft("RGB", (224, 224))
            image = opened.convert("RGB")
        if self.corruption:
            name, severity = self.corruption
            image = apply_fmow_corruption(image, name, severity, self.seed, index)
        else:
            image = TVF.normalize(
                TVF.to_tensor(TVF.resize(image, (224, 224))), MEAN, STD
            )
        return image, int(row["class_id"]), index


def load_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    categories = sorted({row["category"] for row in rows})
    class_ids = {name: index for index, name in enumerate(categories)}
    for row in rows:
        row["class_id"] = class_ids[row["category"]]
    period_counts = {period: sum(row["period"] == period for row in rows) for period in PERIODS}
    if any(count < 2 for count in period_counts.values()):
        raise ValueError(f"Manifest is missing a required chronological window: {period_counts}")
    if len(categories) != 62:
        raise ValueError(f"Expected 62 FMoW classes, found {len(categories)}")
    return rows, categories


def limit_rows(rows, limit, seed):
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    strata = {}
    for row in rows:
        strata.setdefault((row["category"], row["country"]), []).append(row)
    for candidates in strata.values():
        rng.shuffle(candidates)
    selected = []
    while len(selected) < limit:
        progressed = False
        for key in sorted(strata):
            if strata[key] and len(selected) < limit:
                selected.append(strata[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def make_loader(rows, batch_size, workers, corruption=None, seed=42):
    return DataLoader(
        FMoWImages(rows, corruption, seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def apply_fmow_corruption(image, scenario, severity, seed, index):
    """Use ImageNet-C Gaussian levels and project-defined non-noise transforms."""
    if scenario != "gaussian_noise":
        return drift_transform(scenario, severity, seed)(image, index)
    sigma = (0.08, 0.12, 0.18)[severity - 1]
    tensor = TVF.to_tensor(TVF.resize(image.convert("RGB"), (224, 224)))
    generator = torch.Generator().manual_seed((seed + index) % (2**63 - 1))
    noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
    return TVF.normalize((tensor + sigma * noise).clamp(0, 1), MEAN, STD)


def save_corruption_preview(rows, output_path, seed=42, limit=4):
    """Save a label-preservation panel before adaptation begins."""
    examples, seen = [], set()
    for row in rows:
        if row["category"] not in seen:
            examples.append(row)
            seen.add(row["category"])
        if len(examples) == limit:
            break
    conditions = (
        ("clean", None),
        ("Gaussian σ=.08", ("gaussian_noise", 1)),
        ("Gaussian σ=.12", ("gaussian_noise", 2)),
        ("Gaussian σ=.18", ("gaussian_noise", 3)),
        ("sensor noise s3", ("sensor_noise", 3)),
        ("defocus blur s3", ("defocus_blur", 3)),
    )
    figure, axes = plt.subplots(len(examples), len(conditions), figsize=(15, 2.8 * len(examples)))
    axes = np.atleast_2d(axes)
    mean = torch.tensor(MEAN)[:, None, None]
    std = torch.tensor(STD)[:, None, None]
    for row_index, row in enumerate(examples):
        with Image.open(row["local_path"]) as opened:
            opened.draft("RGB", (224, 224))
            image = opened.convert("RGB")
        for column, (title, corruption) in enumerate(conditions):
            tensor = (
                TVF.normalize(TVF.to_tensor(TVF.resize(image, (224, 224))), MEAN, STD)
                if corruption is None
                else apply_fmow_corruption(image, *corruption, seed, row_index)
            )
            display = (tensor * std + mean).clamp(0, 1)
            axes[row_index, column].imshow(TVF.to_pil_image(display))
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(title)
        axes[row_index, 0].set_ylabel(row["category"], rotation=0, labelpad=55, va="center")
    figure.suptitle("FMoW synthetic corruption preview (original labels retained for audit)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def extract(model, loader, device, desc):
    features, labels = [], []
    model.eval()
    for images, targets, _ in tqdm(loader, desc=desc, unit="batch"):
        tokens = model(pixel_values=images.to(device, non_blocking=True)).last_hidden_state
        features.append(tokens[:, 0].float().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def score_predictions(labels, logits):
    values = np.asarray(logits)
    prediction = values.astype(np.int64) if values.ndim == 1 else values.argmax(axis=1)
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(labels, prediction, average="macro", zero_division=0)),
        "per_class_f1": f1_score(labels, prediction, average=None, zero_division=0).tolist(),
        "predictions": prediction.tolist(),
    }


def matched_metric_sample(rows_by_period, features_by_period, labels_by_period, limit, seed):
    rows = [dict(row) for group in rows_by_period.values() for row in group]
    selected = matched_temporal_sample(
        rows, period_key="period", strata_keys=("category", "country"),
        target_per_period=min(limit, *(len(group) for group in rows_by_period.values())),
        seed=seed,
    )
    index_by_period = {
        period: {row["local_path"]: index for index, row in enumerate(group)}
        for period, group in rows_by_period.items()
    }
    features, labels = {}, {}
    for period in rows_by_period:
        indices = np.asarray([
            index_by_period[period][row["local_path"]]
            for row in selected if row["_matched_period"] == period
        ], dtype=np.int64)
        features[period] = features_by_period[period][indices]
        labels[period] = labels_by_period[period][indices]
    return features, labels


def matched_period_sample(splits, limit, seed):
    """Choose equal-size, class-country-matched samples across all periods."""
    rows = [dict(row) for period in PERIODS for row in splits[period]]
    target = min(limit, *(len(splits[period]) for period in PERIODS))
    selected = matched_temporal_sample(
        rows, period_key="period", strata_keys=("category", "country"),
        target_per_period=target, seed=seed,
    )
    return {
        period: [row for row in selected if row["_matched_period"] == period]
        for period in PERIODS
    }


def fdr_bh(p_values):
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for rank in range(len(values), 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def clustered_bootstrap_delta(
    labels, reference_predictions, candidate_predictions, groups,
    repetitions=200, seed=42,
):
    """Paired score-difference intervals resampling geographic UTM clusters."""
    labels = np.asarray(labels)
    reference_predictions = np.asarray(reference_predictions)
    candidate_predictions = np.asarray(candidate_predictions)
    groups = np.asarray(groups)
    if not (len(labels) == len(reference_predictions) == len(candidate_predictions) == len(groups)):
        raise ValueError("labels, predictions, and groups must have equal lengths")
    buckets = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    if len(buckets) < 2:
        raise ValueError("clustered bootstrap needs at least two groups")
    generator = np.random.default_rng(seed)
    differences = {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
    point = {}
    def score(truth, pred):
        return {
            "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(recall_score(truth, pred, average="macro", zero_division=0)),
        }
    ref_point = score(labels, reference_predictions)
    candidate_point = score(labels, candidate_predictions)
    for name in differences:
        point[name] = candidate_point[name] - ref_point[name]
    for _ in range(repetitions):
        sampled = generator.integers(0, len(buckets), size=len(buckets))
        indices = np.concatenate([buckets[index] for index in sampled])
        ref = score(labels[indices], reference_predictions[indices])
        candidate = score(labels[indices], candidate_predictions[indices])
        for name in differences:
            differences[name].append(candidate[name] - ref[name])
    return {
        name: {
            "candidate_minus_reference": point[name],
            "cluster_bootstrap_95_ci": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "repetitions": repetitions,
            "clusters": len(buckets),
        }
        for name, values in differences.items()
    }


class FMoWClassifier(nn.Module):
    def __init__(self, classes, head_weight, head_bias, lora_rank=0):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(BACKBONE)
        width = self.backbone.config.hidden_size
        self.classifier = nn.Linear(width, classes)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.as_tensor(head_weight))
            self.classifier.bias.copy_(torch.as_tensor(head_bias))
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if lora_rank:
            for block in self.backbone.encoder.layer[-3:]:
                attention = block.attention.attention
                attention.query = LoRALinear(attention.query, rank=lora_rank)
                attention.value = LoRALinear(attention.value, rank=lora_rank)
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def forward(self, images, return_tokens=False):
        tokens = self.backbone(pixel_values=images).last_hidden_state
        logits = self.classifier(tokens[:, 0])
        return (logits, tokens) if return_tokens else logits


def mask_from_teacher(images, tokens, classifier_weight, labels, keep_fraction, mode):
    patches = tokens[:, 1:]
    if mode == "random":
        order = torch.rand(patches.shape[:2], device=images.device).argsort(dim=1)
        count = max(1, round(patches.shape[1] * keep_fraction))
        keep = torch.zeros_like(order, dtype=torch.bool)
        keep.scatter_(1, order[:, :count], True)
    else:
        with torch.no_grad():
            saliency = teacher_saliency(patches, classifier_weight, labels, keep_fraction)
        keep = saliency > 0
    side = round(patches.shape[1] ** 0.5)
    if side * side != patches.shape[1]:
        raise ValueError("DINO patch count must form a square grid")
    patch_mask = keep.reshape(-1, 1, side, side).to(images.dtype)
    return images * F.interpolate(patch_mask, size=images.shape[-2:], mode="nearest")


def train_adaptation(model, teacher, loader, device, mode, epochs, lr, keep_fraction, mask_weight, seed):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    history = []
    for epoch in range(epochs):
        model.train()
        model.backbone.eval()
        losses = []
        for images, labels, _ in tqdm(loader, desc=f"{mode} epoch {epoch + 1}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if mode in ("saliency_only", "saliency_lora", "random_mask_only", "random_mask_lora"):
                with torch.no_grad():
                    teacher_tokens = teacher.backbone(pixel_values=images).last_hidden_state
                    predicted = teacher.classifier(teacher_tokens[:, 0]).argmax(1)
                    mask_mode = "random" if mode.startswith("random") else "saliency"
                    masked = mask_from_teacher(
                        images, teacher_tokens, teacher.classifier.weight,
                        predicted, keep_fraction, mask_mode,
                    )
                loss = loss + mask_weight * F.cross_entropy(model(masked), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses))})
    return history


def entropy_tta(head_weight, head_bias, classes, loader, device, rank, seed, lr=1e-5):
    """One label-free online entropy update per test batch, then score that batch."""
    torch.manual_seed(seed)
    model = FMoWClassifier(classes, head_weight, head_bias, lora_rank=rank).to(device).eval()
    for parameter in model.classifier.parameters():
        parameter.requires_grad = False
    parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and (".down." in name or ".up." in name)
    ]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    predictions, targets_all = [], []
    for images, targets, _ in tqdm(loader, desc="label-free entropy TTA", unit="batch"):
        images = images.to(device)
        with torch.enable_grad():
            logits = model(images)
            probabilities = logits.softmax(dim=1)
            entropy = -(probabilities * logits.log_softmax(dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            entropy.backward()
            optimizer.step()
        with torch.no_grad():
            predictions.append(model(images).argmax(1).cpu().numpy())
        targets_all.append(targets.numpy())
    return {
        **score_predictions(np.concatenate(targets_all), np.concatenate(predictions)),
        "protocol": "one unlabeled entropy update on each batch before predicting it",
        "labels_used_for_adaptation": False,
        "adaptation_learning_rate": lr,
    }


@torch.inference_mode()
def evaluate_model(model, loader, device, desc):
    model.eval()
    labels, logits = [], []
    for images, targets, _ in tqdm(loader, desc=desc, unit="batch"):
        logits.append(model(images.to(device)).float().cpu().numpy())
        labels.append(targets.numpy())
    return score_predictions(np.concatenate(labels), np.concatenate(logits))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--adapt-epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--feature-limit", type=int, default=1000)
    parser.add_argument("--stress-limit", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--permutation-limit", type=int, default=200)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--adaptation-limit", type=int)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="load trained adaptation arms and rerun evaluation/stress tests")
    parser.add_argument("--keep-fraction", type=float, default=0.4)
    parser.add_argument("--mask-weight", type=float, default=0.5)
    parser.add_argument("--corruptions", nargs="+", default=("gaussian_noise", "sensor_noise", "defocus_blur"))
    parser.add_argument("--severities", nargs="+", type=int, default=(1, 2, 3))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.adapt_epochs < 1 or args.rank < 1 or not 0 < args.keep_fraction < 1:
        raise ValueError("invalid training settings")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("FMoW pilot requires the server GPU")
    device = "cuda"
    rows, categories = load_manifest(args.manifest)
    splits = {period: [row for row in rows if row["period"] == period] for period in PERIODS}
    splits["adapt_2014_15"] = limit_rows(splits["adapt_2014_15"], args.adaptation_limit, args.seed)
    for name in PERIODS:
        splits[name] = limit_rows(splits[name], None, args.seed)
    loaders = {name: make_loader(splits[name], args.batch_size, args.workers) for name in PERIODS}

    backbone = Dinov2Model.from_pretrained(BACKBONE).to(device).eval()
    historical_features, historical_labels = extract(backbone, loaders[PERIODS[0]], device, "clean historical embeddings")
    adaptation_features, adaptation_labels = extract(backbone, loaders[PERIODS[1]], device, "clean adaptation embeddings")
    future_features, future_labels = extract(backbone, loaders[PERIODS[2]], device, "clean future embeddings")

    metric_n = min(args.feature_limit, *(len(splits[period]) for period in PERIODS))
    clean_metric_features, _ = matched_metric_sample(
        splits,
        dict(zip(PERIODS, (historical_features, adaptation_features, future_features))),
        dict(zip(PERIODS, (historical_labels, adaptation_labels, future_labels))),
        metric_n, args.seed,
    )
    null_rows = splits[PERIODS[0]].copy()
    random.Random(args.seed + 3).shuffle(null_rows)
    null_a_rows, null_b_rows = [], []
    buckets = {}
    for row in null_rows:
        buckets.setdefault((row["category"], row["country"]), []).append(row)
    for values in buckets.values():
        midpoint = len(values) // 2
        null_a_rows.extend(values[:midpoint])
        null_b_rows.extend(values[midpoint:])
    historical_index = {row["local_path"]: index for index, row in enumerate(splits[PERIODS[0]])}
    null_indices = matched_temporal_sample(
        [dict(row, _null_period=name) for name, group in (("null_a", null_a_rows), ("null_b", null_b_rows)) for row in group],
        period_key="_null_period", strata_keys=("category", "country"),
        target_per_period=min(args.feature_limit, len(null_a_rows), len(null_b_rows)),
        seed=args.seed + 4,
    )
    null_a_ix = np.asarray([historical_index[row["local_path"]] for row in null_indices if row["_matched_period"] == "null_a"])
    null_b_ix = np.asarray([historical_index[row["local_path"]] for row in null_indices if row["_matched_period"] == "null_b"])
    null_a, null_b = historical_features[null_a_ix], historical_features[null_b_ix]
    clean_pairs = ((0, 1), (1, 2), (0, 2))
    drift_rows = [
        {"comparison": f"natural_clean_{PERIODS[left]}_vs_{PERIODS[right]}",
         **distribution_shift_metrics(
            clean_metric_features[PERIODS[left]], clean_metric_features[PERIODS[right]],
            permutations=args.permutations, permutation_limit=args.permutation_limit,
            seed=args.seed + left * 11 + right,
        )}
        for left, right in clean_pairs
    ] + [
        {"comparison": "clean_historical_split_half_null", **distribution_shift_metrics(
            null_a, null_b, permutations=args.permutations,
            permutation_limit=args.permutation_limit, seed=args.seed + 3,
        )},
    ]
    q_values = fdr_bh([row["mmd_permutation_p"] for row in drift_rows])
    for row, q_value in zip(drift_rows, q_values):
        row["mmd_bh_q"] = q_value

    probe = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=args.seed,
    ))
    historical_groups = np.asarray([f"{row['country']}:{row['utm']}" for row in splits[PERIODS[0]]])
    fold_count = min(5, len(np.unique(historical_groups)))
    historical_cv = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=args.seed)
    hist_probe_logits = cross_val_predict(
        probe, historical_features, historical_labels, cv=historical_cv,
        groups=historical_groups, method="decision_function",
    )
    probe.fit(historical_features, historical_labels)
    adaptation_probe_logits = probe.decision_function(adaptation_features)
    future_probe_logits = probe.decision_function(future_features)
    if hist_probe_logits.ndim == 1:
        hist_probe_logits = np.stack((-hist_probe_logits, hist_probe_logits), axis=1)
        adaptation_probe_logits = np.stack((-adaptation_probe_logits, adaptation_probe_logits), axis=1)
        future_probe_logits = np.stack((-future_probe_logits, future_probe_logits), axis=1)
    clean_scores = {
        "frozen_historical": {
            "historical_grouped_oof": score_predictions(historical_labels, hist_probe_logits),
            "adaptation": score_predictions(adaptation_labels, adaptation_probe_logits),
            "future": score_predictions(future_labels, future_probe_logits),
        }
    }
    scaler = probe.named_steps["standardscaler"]
    logistic = probe.named_steps["logisticregression"]
    head_weight = logistic.coef_ / scaler.scale_[None, :]
    head_bias = logistic.intercept_ - (scaler.mean_ / scaler.scale_) @ logistic.coef_.T
    if len(head_weight) == 1:
        head_weight = np.concatenate((-head_weight, head_weight), axis=0)
        head_bias = np.concatenate((-head_bias, head_bias), axis=0)
    if head_weight.shape[0] != len(categories):
        raise RuntimeError("linear-probe output dimension does not match FMoW classes")

    args.output.mkdir(parents=True)
    save_corruption_preview(splits[PERIODS[2]], args.output / "corruption_preview.png", args.seed)
    payload = {
        "dataset": "custom original FMoW-RGB chronological subset; not the official FMoW-Time benchmark",
        "feature_extractor": BACKBONE,
        "device": device,
        "seed": args.seed,
        "periods": {name: len(splits[name]) for name in PERIODS},
        "class_country_matched_manifest": str(args.manifest),
        "crossed_stress_design": {
            "periods": PERIODS,
            "scenarios": args.corruptions,
            "severities": args.severities,
            "matched_images_per_period": min(args.stress_limit, *(len(splits[p]) for p in PERIODS)),
            "class_country_composition_matched_across_periods": True,
        },
        "clean_drift_audit": drift_rows,
        "clean_classification": clean_scores,
        "adaptation_arms": {},
        "synthetic_stress": {},
    }
    (args.output / "clean_drift_audit.json").write_text(json.dumps({
        "dataset": payload["dataset"], "periods": payload["periods"],
        "drift_detection": drift_rows, "historical_only_performance": clean_scores,
        "note": "Saved before model adaptation and before synthetic corruption runs.",
    }, indent=2), encoding="utf-8")

    teacher = FMoWClassifier(len(categories), head_weight, head_bias).to(device).eval()
    raw_test = clean_scores["frozen_historical"]["future"]
    payload["adaptation_arms"]["frozen_historical"] = {"future": raw_test}
    for mode in ("lora", "saliency_only", "saliency_lora", "random_mask_only", "random_mask_lora"):
        torch.manual_seed(args.seed)
        student = FMoWClassifier(
            len(categories), head_weight, head_bias,
            lora_rank=args.rank if mode in ("lora", "saliency_lora", "random_mask_lora") else 0,
        ).to(device)
        if args.checkpoint_dir:
            checkpoint = args.checkpoint_dir / f"{mode}.pt"
            student.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            history = [{"loaded_checkpoint": str(checkpoint)}]
        else:
            adaptation_loader = make_loader(splits[PERIODS[1]], args.batch_size, args.workers)
            history = train_adaptation(
                student, teacher, adaptation_loader, device, mode,
                args.adapt_epochs, 1e-3, args.keep_fraction, args.mask_weight, args.seed,
            )
        future = evaluate_model(student, loaders[PERIODS[2]], device, f"{mode} clean future")
        payload["adaptation_arms"][mode] = {
            "history": history, "future": future,
            "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        }
        if not args.checkpoint_dir:
            torch.save(student.state_dict(), args.output / f"{mode}.pt")
        print(json.dumps({"arm": mode, "future": {k: future[k] for k in ("accuracy", "macro_f1", "balanced_accuracy")}}), flush=True)

    payload["adaptation_arms"]["entropy_tta"] = {
        "future": entropy_tta(
            head_weight, head_bias, len(categories), loaders[PERIODS[2]],
            device, args.rank, args.seed,
        )
    }
    print(json.dumps({"arm": "entropy_tta", "future": {
        key: payload["adaptation_arms"]["entropy_tta"]["future"][key]
        for key in ("accuracy", "macro_f1", "balanced_accuracy")
    }}), flush=True)

    future_groups = np.asarray([f"{row['country']}:{row['utm']}" for row in splits[PERIODS[2]]])
    raw_future_predictions = np.asarray(clean_scores["frozen_historical"]["future"]["predictions"])
    payload["clean_future_clustered_deltas"] = {}
    for mode in ("lora", "saliency_only", "saliency_lora", "random_mask_only", "random_mask_lora", "entropy_tta"):
        payload["clean_future_clustered_deltas"][f"{mode}_vs_frozen"] = clustered_bootstrap_delta(
            future_labels, raw_future_predictions,
            payload["adaptation_arms"][mode]["future"]["predictions"],
            future_groups, args.bootstrap_repetitions, args.seed,
        )
    for reference, candidate in (("lora", "saliency_lora"), ("random_mask_lora", "saliency_lora")):
        payload["clean_future_clustered_deltas"][f"{candidate}_vs_{reference}"] = clustered_bootstrap_delta(
            future_labels,
            payload["adaptation_arms"][reference]["future"]["predictions"],
            payload["adaptation_arms"][candidate]["future"]["predictions"],
            future_groups, args.bootstrap_repetitions, args.seed + 1,
        )

    stress_rows_by_period = matched_period_sample(splits, args.stress_limit, args.seed + 10)
    clean_stress_features = {}
    for period in PERIODS:
        period_index = {row["local_path"]: i for i, row in enumerate(splits[period])}
        indices = np.asarray([period_index[row["local_path"]] for row in stress_rows_by_period[period]])
        clean_stress_features[period] = {
            "features": clean_metric_features[period][:0],  # assigned from the full-period arrays below
            "rows": stress_rows_by_period[period],
            "indices": indices,
        }
    full_features = dict(zip(PERIODS, (historical_features, adaptation_features, future_features)))
    for period in PERIODS:
        clean_stress_features[period]["features"] = full_features[period][clean_stress_features[period]["indices"]]

    stress_p = []
    stress_feature_cache = {}
    crossed_grid = crossed_corruption_grid(PERIODS, args.corruptions, args.severities)
    checkpoint_root = args.checkpoint_dir or args.output
    adaptation_modes = ("lora", "saliency_only", "saliency_lora", "random_mask_only", "random_mask_lora")
    for cell in crossed_grid:
        period, scenario, severity = cell["period"], cell["scenario"], cell["severity"]
        rows_for_period = stress_rows_by_period[period]
        corruption = None if scenario == "clean" else (scenario, severity)
        condition_loader = make_loader(
            rows_for_period, args.batch_size, args.workers,
            corruption=corruption, seed=args.seed,
        )
        condition = f"{period}__{scenario}_s{severity}"
        if scenario == "clean":
            condition_features = clean_stress_features[period]["features"]
            feature_shift = None
        else:
            condition_features, _ = extract(backbone, condition_loader, device, f"{condition} embeddings")
            feature_shift = distribution_shift_metrics(
                clean_stress_features[period]["features"], condition_features,
                permutations=args.permutations, permutation_limit=args.permutation_limit,
                seed=args.seed + severity, paired=True,
            )
            stress_p.append((condition, feature_shift["mmd_permutation_p"]))
        stress_feature_cache[(period, scenario, severity)] = condition_features
        result = {
            "period": period, "scenario": scenario, "severity": severity,
            "feature_shift_vs_same_clean_images": feature_shift,
            "frozen_historical": evaluate_model(teacher, condition_loader, device, condition),
        }
        for mode in adaptation_modes:
            model = FMoWClassifier(
                len(categories), head_weight, head_bias,
                lora_rank=args.rank if mode in ("lora", "saliency_lora", "random_mask_lora") else 0,
            ).to(device)
            model.load_state_dict(torch.load(checkpoint_root / f"{mode}.pt", map_location=device, weights_only=True))
            result[mode] = evaluate_model(model, condition_loader, device, f"{condition} {mode}")
            del model
            torch.cuda.empty_cache()
        result["entropy_tta"] = entropy_tta(
            head_weight, head_bias, len(categories), condition_loader,
            device, args.rank, args.seed,
        )
        payload["synthetic_stress"][condition] = result
        print(json.dumps({"condition": condition, "frozen_accuracy": result["frozen_historical"]["accuracy"], "saliency_lora_accuracy": result["saliency_lora"]["accuracy"]}), flush=True)

    stress_q = fdr_bh([p_value for _, p_value in stress_p])
    for (condition, _), q_value in zip(stress_p, stress_q):
        payload["synthetic_stress"][condition]["feature_shift_vs_same_clean_images"]["mmd_bh_q"] = q_value

    payload["progressive_schedules"] = {}
    schedule_p = []
    for schedule_name, severities in (("mild_progression", (0, 1, 2)), ("stronger_future_sensitivity", (0, 1, 3))):
        schedule = progressive_corruption_schedule(PERIODS, "gaussian_noise", severities)
        baseline_features = stress_feature_cache[(PERIODS[0], "clean", 0)]
        cells = []
        for cell in schedule:
            period, scenario, severity = cell["period"], cell["scenario"], cell["severity"]
            condition = f"{period}__{scenario}_s{severity}"
            features = stress_feature_cache[(period, scenario, severity)]
            if period == PERIODS[0]:
                shift_from_initial = None
            else:
                shift_from_initial = distribution_shift_metrics(
                    baseline_features, features,
                    permutations=args.permutations, permutation_limit=args.permutation_limit,
                    seed=args.seed + severity + 100, paired=False,
                )
                schedule_p.append((schedule_name, period, shift_from_initial["mmd_permutation_p"]))
            grid_result = payload["synthetic_stress"][condition]
            stress_labels = np.asarray([row["class_id"] for row in stress_rows_by_period[period]])
            stress_groups = np.asarray([f"{row['country']}:{row['utm']}" for row in stress_rows_by_period[period]])
            cells.append({
                **cell, "condition": condition,
                "shift_vs_initial_clean_historical": shift_from_initial,
                "classification": {
                    mode: {key: grid_result[mode][key] for key in ("accuracy", "macro_f1", "balanced_accuracy")}
                    for mode in ("frozen_historical", *adaptation_modes, "entropy_tta")
                },
                "clustered_score_deltas": {
                    "saliency_lora_vs_lora": clustered_bootstrap_delta(
                        stress_labels,
                        grid_result["lora"]["predictions"],
                        grid_result["saliency_lora"]["predictions"],
                        stress_groups, args.bootstrap_repetitions, args.seed + severity + 1,
                    ),
                    "saliency_only_vs_frozen": clustered_bootstrap_delta(
                        stress_labels,
                        grid_result["frozen_historical"]["predictions"],
                        grid_result["saliency_only"]["predictions"],
                        stress_groups, args.bootstrap_repetitions, args.seed + severity + 2,
                    ),
                },
            })
        payload["progressive_schedules"][schedule_name] = cells
    schedule_q = fdr_bh([p_value for _, _, p_value in schedule_p])
    for (schedule_name, period, _), q_value in zip(schedule_p, schedule_q):
        for cell in payload["progressive_schedules"][schedule_name]:
            if cell["period"] == period:
                cell["shift_vs_initial_clean_historical"]["mmd_bh_q"] = q_value

    (args.output / "fmow_temporal_pilot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {args.output / 'fmow_temporal_pilot.json'}")


if __name__ == "__main__":
    main()
