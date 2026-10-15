"""Measure whether an update reduces condition shift without feature collapse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics


REQUIRED_ARRAYS = ("normal_features", "adverse_features")


def _matrix(values, name):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 5:
        raise ValueError(f"{name} must be a matrix with at least five rows")
    return matrix


def paired_cosine_distance(first, second):
    first = _matrix(first, "first")
    second = _matrix(second, "second")
    if first.shape != second.shape:
        raise ValueError("Paired feature matrices must have the same shape")
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    similarity = np.sum(first * second, axis=1) / np.maximum(denominator, 1e-12)
    return np.clip(1.0 - similarity, 0.0, 2.0)


def unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def row_jensen_shannon(first, second):
    first = _matrix(first, "first probabilities")
    second = _matrix(second, "second probabilities")
    if first.shape != second.shape:
        raise ValueError("Paired probability matrices must have the same shape")
    first = first / np.maximum(first.sum(axis=1, keepdims=True), 1e-12)
    second = second / np.maximum(second.sum(axis=1, keepdims=True), 1e-12)
    midpoint = 0.5 * (first + second)
    first_term = np.where(
        first > 0, first * np.log(first / np.maximum(midpoint, 1e-12)), 0
    )
    second_term = np.where(
        second > 0, second * np.log(second / np.maximum(midpoint, 1e-12)), 0
    )
    return 0.5 * (first_term.sum(axis=1) + second_term.sum(axis=1))


def feature_health(features):
    features = _matrix(features, "features")
    centred = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    energy = singular_values**2
    probabilities = energy / max(float(energy.sum()), 1e-12)
    entropy = -np.sum(
        np.where(probabilities > 0, probabilities * np.log(probabilities), 0)
    )
    return {
        "mean_feature_variance": float(np.var(features, axis=0).mean()),
        "effective_rank": float(np.exp(entropy)),
    }


def bootstrap_mean_change(historical, adapted, seed=42, repetitions=1000):
    historical = np.asarray(historical, dtype=np.float64)
    adapted = np.asarray(adapted, dtype=np.float64)
    if historical.shape != adapted.shape:
        raise ValueError("Paired metric arrays must have the same shape")
    generator = np.random.default_rng(seed)
    changes = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = generator.integers(0, len(historical), len(historical))
        changes[index] = np.mean(historical[sample] - adapted[sample])
    return {
        "historical_minus_adapted_mean": float(np.mean(historical - adapted)),
        "bootstrap_95_ci": [
            float(np.quantile(changes, 0.025)),
            float(np.quantile(changes, 0.975)),
        ],
        "bootstrap_repetitions": repetitions,
    }


def audit_features(arrays, *, permutations=199, permutation_limit=200, seed=42):
    normal = _matrix(arrays["normal_features"], "normal_features")
    adverse = _matrix(arrays["adverse_features"], "adverse_features")
    if normal.shape != adverse.shape:
        raise ValueError("Normal and adverse feature matrices must be paired")
    distances = paired_cosine_distance(normal, adverse)
    return {
        "distribution_shift": distribution_shift_metrics(
            unit_rows(normal),
            unit_rows(adverse),
            permutations=permutations,
            permutation_limit=permutation_limit,
            seed=seed,
        ),
        "paired_feature_cosine_distance": {
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
        },
        "normal_feature_health": feature_health(normal),
        "adverse_feature_health": feature_health(adverse),
        "distribution_feature_scaling": "row-wise L2 normalisation",
    }, distances


def optional_pair_metric(arrays, first_name, second_name):
    if first_name not in arrays or second_name not in arrays:
        return None
    values = row_jensen_shannon(arrays[first_name], arrays[second_name])
    return {"mean": float(values.mean()), "median": float(np.median(values))}, values


def compare_feature_archives(
    historical,
    adapted,
    *,
    saliency_aligned=False,
    permutations=199,
    permutation_limit=200,
    seed=42,
    bootstrap_repetitions=1000,
):
    for name in REQUIRED_ARRAYS:
        if name not in historical or name not in adapted:
            raise KeyError(f"Both archives must contain {name}")

    old_metrics, old_distances = audit_features(
        historical,
        permutations=permutations,
        permutation_limit=permutation_limit,
        seed=seed,
    )
    new_metrics, new_distances = audit_features(
        adapted,
        permutations=permutations,
        permutation_limit=permutation_limit,
        seed=seed,
    )
    payload = {
        "historical": old_metrics,
        "adapted": new_metrics,
        "change": {
            "mmd_squared_reduction": old_metrics["distribution_shift"]["mmd_squared"]
            - new_metrics["distribution_shift"]["mmd_squared"],
            "domain_auc_reduction": old_metrics["distribution_shift"][
                "domain_classifier_roc_auc"
            ]
            - new_metrics["distribution_shift"]["domain_classifier_roc_auc"],
            "paired_feature_distance": bootstrap_mean_change(
                old_distances,
                new_distances,
                seed=seed,
                repetitions=bootstrap_repetitions,
            ),
            "adverse_variance_ratio": new_metrics["adverse_feature_health"][
                "mean_feature_variance"
            ]
            / max(
                old_metrics["adverse_feature_health"]["mean_feature_variance"],
                1e-12,
            ),
            "adverse_effective_rank_ratio": new_metrics["adverse_feature_health"][
                "effective_rank"
            ]
            / max(old_metrics["adverse_feature_health"]["effective_rank"], 1e-12),
        },
    }

    old_prediction = optional_pair_metric(
        historical, "normal_probabilities", "adverse_probabilities"
    )
    new_prediction = optional_pair_metric(
        adapted, "normal_probabilities", "adverse_probabilities"
    )
    if old_prediction and new_prediction:
        old_summary, old_rows = old_prediction
        new_summary, new_rows = new_prediction
        payload["historical"]["paired_prediction_js"] = old_summary
        payload["adapted"]["paired_prediction_js"] = new_summary
        payload["change"]["paired_prediction_js"] = bootstrap_mean_change(
            old_rows,
            new_rows,
            seed=seed,
            repetitions=bootstrap_repetitions,
        )

    has_saliency = all(
        name in historical and name in adapted
        for name in ("normal_saliency", "adverse_saliency")
    )
    if saliency_aligned and has_saliency:
        old_summary, old_rows = optional_pair_metric(
            historical, "normal_saliency", "adverse_saliency"
        )
        new_summary, new_rows = optional_pair_metric(
            adapted, "normal_saliency", "adverse_saliency"
        )
        payload["historical"]["paired_saliency_js"] = old_summary
        payload["adapted"]["paired_saliency_js"] = new_summary
        payload["change"]["paired_saliency_js"] = bootstrap_mean_change(
            old_rows,
            new_rows,
            seed=seed,
            repetitions=bootstrap_repetitions,
        )
    elif has_saliency:
        payload["saliency_note"] = (
            "Saliency arrays were present but were not compared spatially. ACDC "
            "normal/adverse views require correspondence warping and valid-region "
            "masking before patch locations are comparable."
        )
    return payload


def _state_dict(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state.get("model", state) if isinstance(state, dict) else state


def compare_checkpoints(historical_path, adapted_path):
    historical = _state_dict(historical_path)
    adapted = _state_dict(adapted_path)
    common = sorted(set(historical) & set(adapted))
    rows = []
    squared_change = 0.0
    squared_reference = 0.0
    changed = 0
    total = 0
    for name in common:
        first, second = historical[name], adapted[name]
        if (
            not torch.is_tensor(first)
            or not torch.is_tensor(second)
            or first.shape != second.shape
        ):
            continue
        first = first.detach().float()
        second = second.detach().float()
        change = torch.linalg.vector_norm(second - first).item()
        reference = torch.linalg.vector_norm(first).item()
        count = first.numel()
        changed_count = int(torch.count_nonzero((second - first).abs() > 1e-8))
        squared_change += change**2
        squared_reference += reference**2
        changed += changed_count
        total += count
        rows.append(
            {
                "parameter": name,
                "relative_l2_change": change / max(reference, 1e-12),
                "changed_fraction": changed_count / count,
            }
        )
    rows.sort(key=lambda row: row["relative_l2_change"], reverse=True)
    return {
        "common_parameter_tensors": len(rows),
        "relative_l2_change": np.sqrt(squared_change)
        / max(np.sqrt(squared_reference), 1e-12),
        "changed_parameter_fraction": changed / max(total, 1),
        "largest_relative_changes": rows[:20],
        "interpretation": (
            "Parameter movement confirms that optimisation changed the model. It "
            "does not establish useful adaptation without task and representation checks."
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare historical and adapted representations on paired views."
    )
    parser.add_argument("--historical-features", type=Path, required=True)
    parser.add_argument("--adapted-features", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path)
    parser.add_argument("--adapted-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--permutation-limit", type=int, default=200)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--saliency-aligned",
        action="store_true",
        help="Compare saliency maps only after correspondence warping and masking.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if bool(args.historical_checkpoint) != bool(args.adapted_checkpoint):
        raise ValueError("Provide both checkpoints or neither checkpoint")
    with np.load(args.historical_features) as historical, np.load(
        args.adapted_features
    ) as adapted:
        payload = compare_feature_archives(
            historical,
            adapted,
            saliency_aligned=args.saliency_aligned,
            permutations=args.permutations,
            permutation_limit=args.permutation_limit,
            seed=args.seed,
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
    if args.historical_checkpoint:
        payload["parameter_update"] = compare_checkpoints(
            args.historical_checkpoint, args.adapted_checkpoint
        )
    payload["interpretation"] = {
        "positive_reduction": (
            "A positive MMD, domain-AUC, paired-feature, prediction-JS or saliency-JS "
            "reduction means that the adapted model is less condition-sensitive."
        ),
        "claim_boundary": (
            "Reduced discrepancy supports adaptation only when adverse-condition task "
            "performance improves, normal-condition performance is retained, and feature "
            "variance and effective rank do not collapse."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
