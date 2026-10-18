"""Shared classification, temporal, routing, and saliency agreement metrics."""

from collections import defaultdict

import numpy as np


def classification_metrics(truth, prediction, classes=100):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if not len(truth):
        raise ValueError("Cannot score an empty evaluation set")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(classes),
        where=(precision + recall) > 0,
    )
    iou = np.divide(
        tp,
        support + predicted - tp,
        out=np.zeros(classes),
        where=(support + predicted - tp) > 0,
    )
    active = (support + predicted) > 0
    return {
        "count": len(truth),
        "accuracy": round(float(tp.sum() / len(truth)), 4),
        "macro_precision": round(float(precision[active].mean()), 4),
        "macro_recall": round(float(recall[active].mean()), 4),
        "macro_f1": round(float(f1[active].mean()), 4),
        "macro_iou": round(float(iou[active].mean()), 4),
        "weighted_f1": round(float(np.average(f1, weights=support)), 4),
        "per_class": {
            str(index): {
                "support": int(support[index]),
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "iou": round(float(iou[index]), 4),
            }
            for index in range(classes)
            if active[index]
        },
        "confusion_matrix": matrix.tolist(),
    }


def grouped_classification(
    records, truth_key="true_class", prediction_key="prediction"
):
    groups = defaultdict(list)
    for record in records:
        groups["all"].append(record)
        groups[str(record["year"])].append(record)
    return {
        key: classification_metrics(
            [r[truth_key] for r in rows], [r[prediction_key] for r in rows]
        )
        for key, rows in groups.items()
    }


def classwise_model_comparison(
    records,
    prediction_keys=("old_class", "new_class", "routed_class"),
    classes=100,
    min_support=20,
):
    """Compare model variants per class and by historical difficulty.

    Per-class F1 and IoU are one-versus-rest scores. Classes below ``min_support``
    remain visible but are excluded from aggregate improvement claims.
    """
    if not records:
        raise ValueError("Cannot compare models on an empty evaluation set")
    if min_support < 1:
        raise ValueError("min_support must be at least 1")

    truth = np.asarray([row["true_class"] for row in records], dtype=np.int64)
    scores = {
        key: classification_metrics(
            truth,
            [row[key] for row in records],
            classes=classes,
        )
        for key in prediction_keys
    }
    baseline_key = prediction_keys[0]
    support = np.bincount(truth, minlength=classes)
    records_by_class = defaultdict(list)
    for record in records:
        records_by_class[record["true_class"]].append(record)
    rows = []
    for class_id in np.flatnonzero(support):
        class_id = int(class_id)
        class_scores = {
            key.removesuffix("_class"): scores[key]["per_class"][str(class_id)]
            for key in prediction_keys
        }
        baseline = class_scores[baseline_key.removesuffix("_class")]
        row = {
            "class_id": class_id,
            "support": int(support[class_id]),
            "sufficient_support": bool(support[class_id] >= min_support),
            "models": class_scores,
            "f1_delta_pp": {
                key.removesuffix("_class"): round(
                    100
                    * (class_scores[key.removesuffix("_class")]["f1"] - baseline["f1"]),
                    2,
                )
                for key in prediction_keys[1:]
            },
            "iou_delta_pp": {
                key.removesuffix("_class"): round(
                    100
                    * (
                        class_scores[key.removesuffix("_class")]["iou"]
                        - baseline["iou"]
                    ),
                    2,
                )
                for key in prediction_keys[1:]
            },
        }
        if "routed_to_new" in records[0]:
            class_records = records_by_class[class_id]
            row["routing"] = {
                "routed_to_new": sum(bool(r["routed_to_new"]) for r in class_records),
                "route_rate": round(
                    sum(bool(r["routed_to_new"]) for r in class_records)
                    / len(class_records),
                    4,
                ),
                "helpful": sum(
                    r["routed_to_new"]
                    and r[baseline_key] != class_id
                    and r[prediction_keys[1]] == class_id
                    for r in class_records
                ),
                "harmful": sum(
                    r["routed_to_new"]
                    and r[baseline_key] == class_id
                    and r[prediction_keys[1]] != class_id
                    for r in class_records
                ),
                "beneficial_available": sum(
                    r[baseline_key] != class_id and r[prediction_keys[1]] == class_id
                    for r in class_records
                ),
                "beneficial_missed": sum(
                    not r["routed_to_new"]
                    and r[baseline_key] != class_id
                    and r[prediction_keys[1]] == class_id
                    for r in class_records
                ),
                "harmful_available": sum(
                    r[baseline_key] == class_id and r[prediction_keys[1]] != class_id
                    for r in class_records
                ),
                "harmful_avoided": sum(
                    not r["routed_to_new"]
                    and r[baseline_key] == class_id
                    and r[prediction_keys[1]] != class_id
                    for r in class_records
                ),
            }
        rows.append(row)

    eligible = [row for row in rows if row["sufficient_support"]]
    comparisons = {}
    for key in prediction_keys[1:]:
        name = key.removesuffix("_class")
        deltas = np.asarray([row["f1_delta_pp"][name] for row in eligible])
        comparisons[name] = {
            "classes_improved": int(np.sum(deltas > 0)),
            "classes_unchanged": int(np.sum(deltas == 0)),
            "classes_worsened": int(np.sum(deltas < 0)),
            "mean_f1_delta_pp": round(float(deltas.mean()), 2) if len(deltas) else None,
            "median_f1_delta_pp": round(float(np.median(deltas)), 2)
            if len(deltas)
            else None,
        }

    ordered = sorted(
        eligible,
        key=lambda row: (
            row["models"][baseline_key.removesuffix("_class")]["f1"],
            row["class_id"],
        ),
    )
    strata = []
    stratum_count = min(4, len(ordered))
    partitions = np.array_split(ordered, stratum_count) if stratum_count else []
    for index, stratum_rows in enumerate(partitions, 1):
        stratum_rows = list(stratum_rows)
        if not stratum_rows:
            continue
        model_means = {
            key.removesuffix("_class"): round(
                float(
                    np.mean(
                        [
                            row["models"][key.removesuffix("_class")]["f1"]
                            for row in stratum_rows
                        ]
                    )
                ),
                4,
            )
            for key in prediction_keys
        }
        strata.append(
            {
                "stratum": f"Q{index}",
                "description": "weakest historical classes"
                if index == 1
                else "strongest historical classes"
                if index == stratum_count
                else "middle historical classes",
                "class_count": len(stratum_rows),
                "class_ids": [row["class_id"] for row in stratum_rows],
                "historical_f1_range": [
                    min(row["models"]["old"]["f1"] for row in stratum_rows),
                    max(row["models"]["old"]["f1"] for row in stratum_rows),
                ],
                "mean_f1": model_means,
                "mean_f1_delta_pp": {
                    name: round(100 * (value - model_means["old"]), 2)
                    for name, value in model_means.items()
                    if name != "old"
                },
            }
        )

    return {
        "minimum_support": min_support,
        "eligible_classes": len(eligible),
        "excluded_low_support_classes": len(rows) - len(eligible),
        "comparison": comparisons,
        "historical_f1_rank_strata": strata,
        "classes": rows,
    }


def binary_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    return {
        "count": len(truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0.0,
        "iou": round(tp / (tp + fp + fn), 4) if tp + fp + fn else 0.0,
    }


def topk_map_agreement(first, second, fraction=0.2):
    """Overlap of the top-attended patches. Agreement, not localization quality."""
    first = np.asarray(first)
    second = np.asarray(second)
    count = max(1, round(first.size * fraction))
    first_mask = np.zeros(first.size, dtype=bool)
    second_mask = np.zeros(second.size, dtype=bool)
    first_mask[np.argpartition(first, -count)[-count:]] = True
    second_mask[np.argpartition(second, -count)[-count:]] = True
    return binary_metrics(first_mask, second_mask)
