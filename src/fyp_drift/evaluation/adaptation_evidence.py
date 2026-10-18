"""Measure whether an update reduces condition shift without feature collapse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fyp_drift.evaluation.drift_metrics import (
    distribution_shift_metrics,
    median_bandwidth,
)
from fyp_drift.evaluation.metrics import classification_metrics


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


def audit_features(
    arrays,
    *,
    permutations=199,
    permutation_limit=200,
    seed=42,
    bandwidth=None,
):
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
            paired=True,
            bandwidth=bandwidth,
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


def optional_task_metrics(arrays):
    required = (
        "normal_labels",
        "adverse_labels",
        "normal_probabilities",
        "adverse_probabilities",
    )
    if not all(name in arrays for name in required):
        return None
    normal_labels = np.asarray(arrays["normal_labels"], dtype=np.int64)
    adverse_labels = np.asarray(arrays["adverse_labels"], dtype=np.int64)
    normal_probabilities = _matrix(
        arrays["normal_probabilities"], "normal_probabilities"
    )
    adverse_probabilities = _matrix(
        arrays["adverse_probabilities"], "adverse_probabilities"
    )
    if len(normal_labels) != len(normal_probabilities) or len(adverse_labels) != len(
        adverse_probabilities
    ):
        raise ValueError("Task labels and probability rows must have matching lengths")
    classes = normal_probabilities.shape[1]
    if adverse_probabilities.shape[1] != classes:
        raise ValueError("Normal and adverse probabilities must use the same classes")
    return {
        "normal": classification_metrics(
            normal_labels, normal_probabilities.argmax(axis=1), classes
        ),
        "adverse": classification_metrics(
            adverse_labels, adverse_probabilities.argmax(axis=1), classes
        ),
    }


def evidence_conclusions(payload, *, alpha, collapse_ratio_floor, clean_tolerance):
    historical_shift = payload["historical"]["distribution_shift"]
    adapted_shift = payload["adapted"]["distribution_shift"]
    change = payload["change"]
    distance_ci = change["paired_feature_distance"]["bootstrap_95_ci"]
    baseline_detected = historical_shift["mmd_permutation_p"] <= alpha
    residual_detected = adapted_shift["mmd_permutation_p"] <= alpha
    representation_supported = all(
        (
            change["mmd_squared_reduction"] > 0,
            change["domain_auc_reduction"] > 0,
            distance_ci[0] > 0,
            change["adverse_variance_ratio"] >= collapse_ratio_floor,
            change["adverse_effective_rank_ratio"] >= collapse_ratio_floor,
        )
    )

    conclusions = {
        "baseline_drift": {
            "detected": baseline_detected,
            "test": "RBF MMD permutation test",
            "alpha": alpha,
            "p_value": historical_shift["mmd_permutation_p"],
            "domain_classifier_roc_auc": historical_shift[
                "domain_classifier_roc_auc"
            ],
        },
        "representation_mitigation": {
            "supported": baseline_detected and representation_supported,
            "collapse_ratio_floor": collapse_ratio_floor,
            "criteria": {
                "mmd_reduced": change["mmd_squared_reduction"] > 0,
                "domain_separability_reduced": change["domain_auc_reduction"] > 0,
                "paired_distance_ci_above_zero": distance_ci[0] > 0,
                "variance_retained": change["adverse_variance_ratio"]
                >= collapse_ratio_floor,
                "effective_rank_retained": change["adverse_effective_rank_ratio"]
                >= collapse_ratio_floor,
            },
        },
        "residual_drift": {
            "detected": residual_detected,
            "test": "RBF MMD permutation test on adapted representations",
            "alpha": alpha,
            "p_value": adapted_shift["mmd_permutation_p"],
        },
        "task_mitigation": {"status": "not_assessed"},
    }

    if "task_performance" in payload["historical"]:
        old_task = payload["historical"]["task_performance"]
        new_task = payload["adapted"]["task_performance"]
        adverse_gain = (
            new_task["adverse"]["macro_f1"] - old_task["adverse"]["macro_f1"]
        )
        normal_change = (
            new_task["normal"]["macro_f1"] - old_task["normal"]["macro_f1"]
        )
        conclusions["task_mitigation"] = {
            "status": "supported"
            if adverse_gain > 0 and normal_change >= -clean_tolerance
            else "not_supported",
            "adverse_macro_f1_change": adverse_gain,
            "normal_macro_f1_change": normal_change,
            "normal_retention_tolerance": clean_tolerance,
        }

    task_status = conclusions["task_mitigation"]["status"]
    if not baseline_detected:
        overall = "baseline_drift_not_verified"
    elif not representation_supported:
        overall = "mitigation_not_supported"
    elif task_status == "not_supported":
        overall = "representation_improved_but_task_evidence_failed"
    elif task_status == "not_assessed":
        overall = "representation_mitigation_supported_task_not_assessed"
    elif residual_detected:
        overall = "partial_mitigation_supported_with_residual_drift"
    else:
        overall = "mitigation_supported_no_residual_drift_detected"
    conclusions["overall"] = overall
    return conclusions


def compare_feature_archives(
    historical,
    adapted,
    *,
    saliency_aligned=False,
    permutations=199,
    permutation_limit=200,
    seed=42,
    bootstrap_repetitions=1000,
    alpha=0.05,
    collapse_ratio_floor=0.8,
    clean_tolerance=0.01,
):
    for name in REQUIRED_ARRAYS:
        if name not in historical or name not in adapted:
            raise KeyError(f"Both archives must contain {name}")

    pooled_normal = np.concatenate(
        (
            unit_rows(historical["normal_features"]),
            unit_rows(adapted["normal_features"]),
        )
    )
    pooled_adverse = np.concatenate(
        (
            unit_rows(historical["adverse_features"]),
            unit_rows(adapted["adverse_features"]),
        )
    )
    common_bandwidth = median_bandwidth(pooled_normal, pooled_adverse)

    old_metrics, old_distances = audit_features(
        historical,
        permutations=permutations,
        permutation_limit=permutation_limit,
        seed=seed,
        bandwidth=common_bandwidth,
    )
    new_metrics, new_distances = audit_features(
        adapted,
        permutations=permutations,
        permutation_limit=permutation_limit,
        seed=seed,
        bandwidth=common_bandwidth,
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

    old_task = optional_task_metrics(historical)
    new_task = optional_task_metrics(adapted)
    if bool(old_task) != bool(new_task):
        raise ValueError("Task labels and probabilities must be present in both archives")
    if old_task and new_task:
        payload["historical"]["task_performance"] = old_task
        payload["adapted"]["task_performance"] = new_task

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
    payload["conclusions"] = evidence_conclusions(
        payload,
        alpha=alpha,
        collapse_ratio_floor=collapse_ratio_floor,
        clean_tolerance=clean_tolerance,
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
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--collapse-ratio-floor", type=float, default=0.8)
    parser.add_argument("--clean-retention-tolerance", type=float, default=0.01)
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
            alpha=args.alpha,
            collapse_ratio_floor=args.collapse_ratio_floor,
            clean_tolerance=args.clean_retention_tolerance,
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
