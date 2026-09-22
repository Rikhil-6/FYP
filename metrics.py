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
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(classes), where=(precision + recall) > 0)
    iou = np.divide(tp, support + predicted - tp,
                    out=np.zeros(classes), where=(support + predicted - tp) > 0)
    active = (support + predicted) > 0
    return {
        "count": int(len(truth)),
        "accuracy": round(float(tp.sum() / len(truth)), 4),
        "macro_precision": round(float(precision[active].mean()), 4),
        "macro_recall": round(float(recall[active].mean()), 4),
        "macro_f1": round(float(f1[active].mean()), 4),
        "macro_iou": round(float(iou[active].mean()), 4),
        "weighted_f1": round(float(np.average(f1, weights=support)), 4),
        "per_class": {str(index): {"support": int(support[index]),
                                   "precision": round(float(precision[index]), 4),
                                   "recall": round(float(recall[index]), 4),
                                   "f1": round(float(f1[index]), 4),
                                   "iou": round(float(iou[index]), 4)}
                      for index in range(classes) if active[index]},
        "confusion_matrix": matrix.tolist(),
    }


def grouped_classification(records, truth_key="true_class", prediction_key="prediction"):
    groups = defaultdict(list)
    for record in records:
        groups["all"].append(record)
        groups[str(record["year"])].append(record)
    return {key: classification_metrics([r[truth_key] for r in rows],
                                        [r[prediction_key] for r in rows])
            for key, rows in groups.items()}


def binary_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    return {"count": int(len(truth)), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0.0,
            "iou": round(tp / (tp + fp + fn), 4) if tp + fp + fn else 0.0}


def topk_map_agreement(first, second, fraction=0.2):
    """Overlap of the top-attended patches. Agreement, not localization quality."""
    first = np.asarray(first)
    second = np.asarray(second)
    count = max(1, int(round(first.size * fraction)))
    first_mask = np.zeros(first.size, dtype=bool)
    second_mask = np.zeros(second.size, dtype=bool)
    first_mask[np.argpartition(first, -count)[-count:]] = True
    second_mask[np.argpartition(second, -count)[-count:]] = True
    return binary_metrics(first_mask, second_mask)
