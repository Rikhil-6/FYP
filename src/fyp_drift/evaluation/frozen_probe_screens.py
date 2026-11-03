"""Frozen-backbone screens: does real shift harm a linear probe, across datasets and architectures?

For each (dataset, backbone): extract mean patch features from a frozen backbone, fit a
class-weighted linear head on the reference window, pick its checkpoint on a reference
validation set, and report
  - scores per window,
  - the reference-minus-target accuracy gap with a location-clustered 95% interval,
  - MMD / domain AUC between reference and target features (with a split-half null),
  - scores on real acquisition strata of the target window.

Datasets
  fmow      historical 2010-13 -> future 2016-17 (study splits); strata from recorded
            cloud cover, off-nadir angle and sun elevation.
  iwildcam  WILDS train cameras -> unseen test cameras; strata from the recorded capture
            hour (night 19:00-06:00, when camera traps switch to infrared) vs day.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_CACHE, DEFAULT_MANIFEST, cluster_bootstrap, fit_linear_head, image_cache, load_rows, log, scores,
    study_splits, write_json,
)

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
BACKBONES = {
    "dinov2-small": ("facebook/dinov2-small", "Dinov2Model", IMAGENET, True),
    "dinov2-base": ("facebook/dinov2-base", "Dinov2Model", IMAGENET, True),
    "dinov2-large": ("facebook/dinov2-large", "Dinov2Model", IMAGENET, True),
    "clip-b16": ("openai/clip-vit-base-patch16", "CLIPVisionModel", ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)), True),
    "siglip-b16": ("google/siglip-base-patch16-224", "SiglipVisionModel", ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), False),
}
IWILDCAM = Path("/mnt/sdd/FYP/data/wilds/iwildcam_v2.0")
IWILDCAM_CACHE = Path("/mnt/sdd/FYP/cache/iwildcam_u8_224.npy")


# --------------------------------------------------------------------------- datasets
def fmow_windows():
    """Rows, class count, cache and the four windows; row['idx'] indexes the cache."""
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    windows = {"fit": splits["hist_train"], "select": splits["hist_val"], "reference": splits["hist_val"], "target": splits["future_test"]}
    return rows, classes, DEFAULT_CACHE, windows


def iwildcam_windows(seed=42):
    """A seeded subsample of WILDS iWildCam: train cameras for fitting, unseen cameras as the target."""
    with open(IWILDCAM / "metadata.csv", newline="") as handle:
        everything = list(csv.DictReader(handle))
    rng = random.Random(seed)
    sizes = {"train": 15000, "id_val": 2000, "id_test": 4000, "test": 6000}
    picked = {split: rng.sample([r for r in everything if r["split"] == split], sizes[split]) for split in sizes}
    rows = [r for split in sizes for r in picked[split]]
    labels = sorted({int(r["y"]) for r in rows})
    remap = {y: i for i, y in enumerate(labels)}
    for i, row in enumerate(rows):
        hour = int(row["datetime"][11:13])
        row.update(idx=i, class_id=remap[int(row["y"])], group=row["location_remapped"],
                   local_path=str(IWILDCAM / "train" / row["filename"]), night=hour >= 19 or hour < 6)
    windows = {"fit": picked["train"], "select": picked["id_val"], "reference": picked["id_test"], "target": picked["test"]}
    return rows, len(labels), IWILDCAM_CACHE, windows


def target_strata(dataset, target):
    if dataset == "iwildcam":
        night = np.array([r["night"] for r in target])
        return {"night": night, "day": ~night}
    cloud = np.array([r["cloud_cover"] for r in target])
    nadir = np.array([r["off_nadir_angle"] for r in target])
    sun = np.array([r["sun_elevation"] for r in target])
    return {"cloud_cover_ge10": cloud >= 10, "cloud_free": cloud == 0,
            "off_nadir_top25": nadir >= np.quantile(nadir, 0.75), "off_nadir_bottom25": nadir <= np.quantile(nadir, 0.25),
            "low_sun_bottom25": sun <= np.quantile(sun, 0.25), "high_sun_top25": sun >= np.quantile(sun, 0.75)}


# --------------------------------------------------------------------------- features
def frozen_mean_patch_features(name, cache, rows, batch_size=64, workers=2):
    import transformers
    hub_id, class_name, (mean, std), has_cls = BACKBONES[name]
    model = getattr(transformers, class_name).from_pretrained(hub_id).cuda().eval()
    mean, std = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]

    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            pixels = torch.from_numpy(np.array(cache[rows[i]["idx"]])).permute(2, 0, 1).float() / 255.0
            return (pixels - mean) / std

    features = []
    with torch.no_grad():
        for images in torch.utils.data.DataLoader(Images(), batch_size=batch_size, num_workers=workers):
            with torch.autocast("cuda", dtype=torch.float16):
                tokens = model(pixel_values=images.cuda(non_blocking=True)).last_hidden_state
            features.append((tokens[:, 1:] if has_cls else tokens).float().mean(1).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(features).numpy()


# --------------------------------------------------------------------------- screen
def accuracy_gap(reference_correct, reference_groups, target_correct, target_groups, repetitions=1000):
    """Reference minus target accuracy, resampling location groups independently on each side."""
    rng = np.random.default_rng(0)
    sides = [(np.unique(g), {k: c[g == k] for k in np.unique(g)}) for c, g in
             ((reference_correct, reference_groups), (target_correct, target_groups))]
    draws = []
    for _ in range(repetitions):
        means = [np.concatenate([by[k] for k in rng.choice(unique, len(unique))]).mean() for unique, by in sides]
        draws.append(means[0] - means[1])
    return {"mean": float(np.mean(draws)), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def drift(features, windows, permutations):
    from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
    rng = np.random.default_rng(42)
    reference = features[rng.choice(np.array([r["idx"] for r in windows["fit"]]), 2000, replace=False)]
    target = features[rng.choice(np.array([r["idx"] for r in windows["target"]]), 1000, replace=False)]
    results = {}
    for name, (a, b) in {"reference_vs_target": (reference[:1000], target), "split_half_null": (reference[:1000], reference[1000:])}.items():
        m = distribution_shift_metrics(a, b, permutations=permutations, permutation_limit=1000, seed=42)
        results[name] = {k: m[k] for k in ("mmd_squared", "mmd_permutation_p", "domain_classifier_roc_auc")}
    return results


def screen(dataset, backbone, rows, classes, cache, windows, output, permutations):
    path = output / f"{dataset}_{backbone}.json"
    if path.exists():
        return json.loads(path.read_text())
    features = frozen_mean_patch_features(backbone, cache, rows)
    index = {k: np.array([r["idx"] for r in v]) for k, v in windows.items()}
    labels = {k: np.array([r["class_id"] for r in v]) for k, v in windows.items()}
    head, select_f1 = fit_linear_head(features[index["fit"]], labels["fit"], classes, features[index["select"]], labels["select"])
    with torch.no_grad():
        predicted = {k: head(torch.tensor(features[index[k]]).cuda()).argmax(1).cpu().numpy() for k in index}
    correct = {k: predicted[k] == labels[k] for k in index}
    groups = {k: np.array([r["group"] for r in v]) for k, v in windows.items()}
    strata = target_strata(dataset, windows["target"])
    result = {"dataset": dataset, "backbone": backbone, "select_macro_f1": select_f1,
              "windows": {k: scores(labels[k], predicted[k], classes) for k in ("select", "reference", "target")},
              "gap_reference_minus_target": accuracy_gap(correct["reference"].astype(float), groups["reference"],
                                                         correct["target"].astype(float), groups["target"]),
              "drift": drift(features, windows, permutations),
              "target_strata": {name: {"n": int(mask.sum()), **scores(labels["target"][mask], predicted["target"][mask], classes)}
                                for name, mask in strata.items()}}
    write_json(path, result)
    log(dataset, backbone, "ref", round(result["windows"]["reference"]["accuracy"], 4), "target", round(result["windows"]["target"]["accuracy"], 4),
        "gap", result["gap_reference_minus_target"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("/mnt/sdd/FYP/runs/today/screens"))
    parser.add_argument("--datasets", nargs="+", default=["fmow", "iwildcam"])
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    parser.add_argument("--permutations", type=int, default=499)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for dataset in args.datasets:
        rows, classes, cache_path, windows = fmow_windows() if dataset == "fmow" else iwildcam_windows()
        cache = image_cache(rows, cache_path, workers=4)
        for backbone in args.backbones:
            r = screen(dataset, backbone, rows, classes, cache, windows, args.output, args.permutations)
            summary.append({"dataset": dataset, "backbone": backbone,
                            "reference_acc": r["windows"]["reference"]["accuracy"], "target_acc": r["windows"]["target"]["accuracy"],
                            "reference_f1": r["windows"]["reference"]["macro_f1"], "target_f1": r["windows"]["target"]["macro_f1"],
                            "gap": r["gap_reference_minus_target"]["mean"], "gap_lo": r["gap_reference_minus_target"]["ci95"][0],
                            "gap_hi": r["gap_reference_minus_target"]["ci95"][1],
                            "mmd_p": r["drift"]["reference_vs_target"]["mmd_permutation_p"],
                            "auc": r["drift"]["reference_vs_target"]["domain_classifier_roc_auc"],
                            **{f"{k}_acc": v["accuracy"] for k, v in r["target_strata"].items()}})
            with open(args.output / "summary.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({k for s in summary for k in s}, key=lambda k: (k not in ("dataset", "backbone"), k)))
                writer.writeheader()
                writer.writerows(summary)


if __name__ == "__main__":
    main()
