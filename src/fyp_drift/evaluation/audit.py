"""Audit when the drift-adapted ViT corrects or harms historical predictions.

The audit keeps sample identities, stratifies old/new/router outcomes, and tests
whether deleting the most salient patches changes true-class confidence more
than deleting an equal number of least salient patches.  This is a bounded
faithfulness check, not proof that a saliency map is an object segmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from fyp_drift.data.clear100 import MEAN, STD, balanced_indices, open_dataset
from fyp_drift.data.corruptions import SCENARIOS, drift_transform, scenario_components
from fyp_drift.models.routing import FEATURES, apply_policy
from fyp_drift.models.siamese import ViTTwinRouter
from fyp_drift.models.vision_transformer import TemporalViT

SALIENCY_OVERLAY_STRENGTH = 0.58


def figure_font(size=14, bold=False):
    names = (
        ("timesbd.ttf", "DejaVuSerif-Bold.ttf")
        if bold
        else ("times.ttf", "DejaVuSerif.ttf")
    )
    roots = (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"))
    for root, name in zip(roots, names):
        candidate = root / name
        if candidate.is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


class IndexedDataset(Dataset):
    def __init__(self, subset, indices):
        self.subset = subset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        index = self.indices[offset]
        image, label, metadata = self.subset[index]
        return image, label, metadata, index


def rgb(tensor):
    mean = torch.tensor(MEAN)[:, None, None]
    std = torch.tensor(STD)[:, None, None]
    value = tensor.detach().cpu() * std + mean
    return Image.fromarray(
        (value.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )


def normalized_saliency(weights):
    side = round(weights.numel() ** 0.5)
    heat = weights.detach().cpu().reshape(side, side).numpy().astype(np.float32)
    low, high = np.quantile(heat, (0.05, 0.95))
    return np.clip((heat - low) / max(float(high - low), 1e-8), 0, 1)


def color_map(values, size, name):
    if name == "inferno":
        positions = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        colors = np.array(
            [(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 8), (252, 255, 164)]
        )
    elif name == "coolwarm":
        positions = np.array([0.0, 0.50, 1.0])
        colors = np.array([(59, 76, 192), (221, 221, 221), (180, 4, 38)])
    else:
        raise ValueError(f"Unknown colour map: {name}")
    rgb_values = np.stack(
        [np.interp(values, positions, colors[:, channel]) for channel in range(3)],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb_values).resize(size, Image.Resampling.BILINEAR)


def overlay(base, weights):
    """Render a robustly normalised heatmap strongly enough to inspect spatial evidence."""
    heat = color_map(normalized_saliency(weights), base.size, "inferno")
    return Image.blend(base, heat, SALIENCY_OVERLAY_STRENGTH)


def saliency_change(old_weights, new_weights, size):
    """Show signed patch-evidence redistribution with a symmetric per-sample scale."""
    side = round(old_weights.numel() ** 0.5)
    old = old_weights.detach().cpu().reshape(side, side).numpy().astype(np.float32)
    new = new_weights.detach().cpu().reshape(side, side).numpy().astype(np.float32)
    difference = new - old
    scale = max(float(np.quantile(np.abs(difference), 0.95)), 1e-8)
    signed = np.clip(0.5 + 0.5 * difference / scale, 0, 1)
    return color_map(signed, size, "coolwarm")


def delete_patches(image, saliency, highest=True, fraction=0.20):
    """Replace an equal number of highest/lowest-saliency patches with mean pixels."""
    patch_side = round(saliency.numel() ** 0.5)
    count = max(1, round(saliency.numel() * fraction))
    order = saliency.argsort(descending=highest)[:count]
    mask = torch.zeros_like(saliency, dtype=torch.float32)
    mask[order] = 1
    mask = mask.reshape(1, 1, patch_side, patch_side)
    mask = torch.nn.functional.interpolate(mask, size=image.shape[-2:], mode="nearest")[
        0
    ].bool()
    changed = image.clone()
    changed[mask.expand_as(changed)] = 0.0  # zero is the normalized dataset mean
    return changed


@torch.no_grad()
def deletion_metrics(model, image, saliency, label):
    base_probability = model(image[None])[0].softmax(1)[0, label]
    salient = delete_patches(image, saliency, highest=True)
    context = delete_patches(image, saliency, highest=False)
    salient_probability = model(salient[None])[0].softmax(1)[0, label]
    context_probability = model(context[None])[0].softmax(1)[0, label]
    salient_drop = float(base_probability - salient_probability)
    context_drop = float(base_probability - context_probability)
    return (
        {
            "true_class_probability": float(base_probability),
            "salient_deletion_drop": salient_drop,
            "context_deletion_drop": context_drop,
            "faithfulness_margin": salient_drop - context_drop,
        },
        salient,
        context,
    )


def outcome(row):
    old_correct = row["old_class"] == row["true_class"]
    new_correct = row["new_class"] == row["true_class"]
    routed = row["routed_to_new"]
    if not old_correct and new_correct:
        return "helpful_routed" if routed else "missed_helpful"
    if old_correct and not new_correct:
        return "harmful_routed" if routed else "protected_old"
    if old_correct and new_correct:
        return "both_correct"
    return "both_wrong"


def save_panel(
    path, clean_image, drift_image, comparison, new_salient, new_context, row
):
    drift_rgb = rgb(drift_image)
    panels = [
        rgb(clean_image),
        drift_rgb,
        overlay(drift_rgb, comparison["old_map"][0]),
        overlay(drift_rgb, comparison["new_map"][0]),
        saliency_change(
            comparison["old_map"][0], comparison["new_map"][0], drift_rgb.size
        ),
        rgb(new_salient),
        rgb(new_context),
    ]
    shifted_name = row["scenario"].replace("_", " ")
    if row.get("drift_components"):
        shifted_name += ": " + row["drift_components"].replace("+", ", ")
    names = (
        "(a) Clean",
        f"(b) {shifted_name}",
        "(c) Historical saliency",
        "(d) Updated saliency",
        "(e) Change: blue old / red new",
        "(f) Remove top 20%",
        "(g) Remove bottom 20%",
    )
    canvas = Image.new("RGB", (224 * len(panels), 316), "white")
    draw = ImageDraw.Draw(canvas)
    for position, (panel, name) in enumerate(zip(panels, names)):
        canvas.paste(panel, (position * 224, 30))
        draw.text((position * 224 + 6, 7), name, font=figure_font(13), fill="black")
    summary = (
        f"{row['outcome']} | sample {row['dataset_index']} | true {row['true_class']} | "
        f"old {row['old_class']} | new {row['new_class']} | routed {row['routed_class']} | "
        f"p(route) {row['route_probability']:.3f} | JS {row['saliency_js']:.4f} | "
        f"new deletion margin {row['new_faithfulness_margin']:+.3f}"
    )
    draw.line((0, 258, canvas.width, 258), fill="#BBBBBB", width=1)
    draw.text((6, 267), summary, font=figure_font(13), fill="black")
    canvas.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument("--scenario", choices=SCENARIOS, default="realistic_mix")
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_state = torch.load(
        args.base_checkpoint, map_location=device, weights_only=True
    )
    classes = base_state["classifier.weight"].shape[0]
    old = TemporalViT("xai", classes).to(device)
    new = TemporalViT("xai", classes).to(device)
    old.load_state_dict(base_state)
    new.load_state_dict(
        torch.load(
            args.run_dir / "models" / "best.pt", map_location=device, weights_only=True
        )
    )
    old.eval()
    new.eval()
    router = ViTTwinRouter(old, new).eval()
    policy = json.loads((args.run_dir / "metrics" / "router_policy.json").read_text())

    dataset = open_dataset(args.data_root)
    clean = dataset.get_subset(
        "test_source", transform=drift_transform("clean", 1, args.seed)
    )
    shifted = dataset.get_subset(
        "test_source",
        transform=drift_transform(args.scenario, args.severity, args.seed),
    )
    indices = torch.where(shifted.metadata_array[:, 0] == args.year - 2002)[0].tolist()
    if args.limit:
        indices = balanced_indices(
            shifted.rows,
            indices,
            min(args.limit, len(indices)),
            args.seed,
        )
    loader = DataLoader(IndexedDataset(shifted, indices), batch_size=args.batch_size)

    rows = []
    with torch.no_grad():
        for images, labels, _, batch_indices in tqdm(
            loader, desc="counterexample audit", unit="batch"
        ):
            comparison = router.compare(images.to(device))
            routed, needs_update, use_new, probabilities, features = apply_policy(
                comparison, policy
            )
            old_class = comparison["old_logits"].argmax(1)
            new_class = comparison["new_logits"].argmax(1)
            old_conf = comparison["old_logits"].softmax(1).max(1).values
            new_conf = comparison["new_logits"].softmax(1).max(1).values
            for position in range(len(labels)):
                row = {
                    "dataset_index": int(batch_indices[position]),
                    "year": args.year,
                    "scenario": args.scenario,
                    "severity": args.severity,
                    "drift_components": "+".join(
                        scenario_components(
                            args.scenario,
                            int(batch_indices[position]),
                            args.severity,
                            args.seed,
                        )
                    ),
                    "true_class": int(labels[position]),
                    "old_class": int(old_class[position]),
                    "new_class": int(new_class[position]),
                    "routed_class": int(routed[position]),
                    "old_confidence": float(old_conf[position]),
                    "new_confidence": float(new_conf[position]),
                    "saliency_js": float(comparison["saliency_gap"][position]),
                    "route_probability": float(probabilities[position]),
                    "needs_update": bool(needs_update[position]),
                    "routed_to_new": bool(use_new[position]),
                    "old_evidence": float(comparison["old_evidence"][position]),
                    "new_evidence": float(comparison["new_evidence"][position]),
                }
                row.update(
                    {
                        name: float(features[position, feature_index])
                        for feature_index, name in enumerate(FEATURES)
                    }
                )
                row["outcome"] = outcome(row)
                rows.append(row)

    groups = {
        name: [row for row in rows if row["outcome"] == name]
        for name in (
            "helpful_routed",
            "missed_helpful",
            "harmful_routed",
            "protected_old",
            "both_correct",
            "both_wrong",
        )
    }
    args.output.mkdir(parents=True, exist_ok=True)
    audited = (
        groups["helpful_routed"]
        + groups["missed_helpful"]
        + groups["harmful_routed"]
        + groups["protected_old"]
    )
    for row in tqdm(audited, desc="faithfulness audit", unit="image"):
        index = row["dataset_index"]
        drift_image, label, _ = shifted[index]
        drift_device = drift_image.to(device)
        with torch.no_grad():
            comparison = router.compare(drift_device[None])
            old_metrics, _, _ = deletion_metrics(
                old, drift_device, comparison["old_map"][0], int(label)
            )
            new_metrics, new_salient, new_context = deletion_metrics(
                new, drift_device, comparison["new_map"][0], int(label)
            )
        row.update({f"old_{key}": value for key, value in old_metrics.items()})
        row.update({f"new_{key}": value for key, value in new_metrics.items()})

    selected = []
    for name in ("helpful_routed", "missed_helpful", "harmful_routed"):
        # Prefer clear, high-probability decisions, then examples whose own
        # saliency deletion response is measurable. Missed corrections are
        # ordered from the cases closest to the route threshold.
        ranked = sorted(
            groups[name],
            key=lambda row: (
                row["route_probability"],
                row["new_faithfulness_margin"],
                row["saliency_js"],
            ),
            reverse=name != "missed_helpful",
        )
        selected.extend(ranked[: args.top_k])

    for row in tqdm(selected, desc="counterexample panels", unit="image"):
        index = row["dataset_index"]
        clean_image, _, _ = clean[index]
        drift_image, _, _ = shifted[index]
        drift_device = drift_image.to(device)
        with torch.no_grad():
            comparison = router.compare(drift_device[None])
            _, new_salient, new_context = deletion_metrics(
                new, drift_device, comparison["new_map"][0], row["true_class"]
            )
        folder = args.output / row["outcome"]
        folder.mkdir(exist_ok=True)
        image_path = folder / f"sample_{index:05d}.png"
        save_panel(
            image_path,
            clean_image,
            drift_image,
            comparison,
            new_salient.cpu(),
            new_context.cpu(),
            row,
        )
        row["panel"] = image_path.relative_to(args.output).as_posix()

    fieldnames = sorted(
        {key for row in rows for key in row} | {key for row in selected for key in row}
    )
    with (args.output / "all_decisions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "selected_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    faithfulness = {}
    for name in ("helpful_routed", "missed_helpful", "harmful_routed", "protected_old"):
        values = groups[name]
        faithfulness[name] = {
            "samples": len(values),
            "mean_old_margin": float(
                np.mean([row["old_faithfulness_margin"] for row in values])
            ),
            "mean_new_margin": float(
                np.mean([row["new_faithfulness_margin"] for row in values])
            ),
            "new_positive_margin_fraction": float(
                np.mean([row["new_faithfulness_margin"] > 0 for row in values])
            ),
        }
    summary = {
        "scope": {
            "year": args.year,
            "scenario": args.scenario,
            "severity": args.severity,
            "samples": len(rows),
            "device": device,
        },
        "counts": {name: len(group) for name, group in groups.items()},
        "router": {
            "helpful_routes": len(groups["helpful_routed"]),
            "harmful_routes": len(groups["harmful_routed"]),
            "missed_helpful": len(groups["missed_helpful"]),
            "net_corrections": len(groups["helpful_routed"])
            - len(groups["harmful_routed"]),
        },
        "faithfulness_by_outcome": faithfulness,
        "selected_cases": selected,
        "interpretation_limit": (
            "Deletion margins test local prediction sensitivity. They do not establish object localization "
            "or prove that saliency caused the model update."
        ),
    }
    (args.output / "counterexample_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["router"] | summary["counts"], indent=2))


if __name__ == "__main__":
    main()
