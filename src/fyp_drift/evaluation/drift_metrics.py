"""Distribution-level metrics for verifying image drift in frozen features."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# The default of 499 randomisations gives a minimum p-value of 0.002 while
# keeping repeated drift audits practical on the project server.
DEFAULT_PERMUTATIONS = 499
PERMUTATION_BATCH_SIZE = 256


def _as_matrix(values):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("Each feature sample must be a matrix with at least two rows")
    return matrix


def squared_distances(first, second):
    first = _as_matrix(first)
    second = _as_matrix(second)
    distances = (
        np.sum(first * first, axis=1)[:, None]
        + np.sum(second * second, axis=1)[None, :]
        - 2 * first @ second.T
    )
    return np.maximum(distances, 0.0)


def median_bandwidth(first, second):
    combined = np.concatenate((_as_matrix(first), _as_matrix(second)), axis=0)
    distances = squared_distances(combined, combined)
    positive = distances[np.triu_indices_from(distances, k=1)]
    positive = positive[positive > 0]
    if not len(positive):
        return 1.0
    return float(np.sqrt(np.median(positive)))


def rbf_kernel(first, second, bandwidth):
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    return np.exp(-squared_distances(first, second) / (2 * bandwidth**2))


def maximum_mean_discrepancy(first, second, bandwidth=None):
    """Return the non-negative, biased RBF MMD squared estimate."""
    first, second = _as_matrix(first), _as_matrix(second)
    bandwidth = bandwidth or median_bandwidth(first, second)
    value = (
        rbf_kernel(first, first, bandwidth).mean()
        + rbf_kernel(second, second, bandwidth).mean()
        - 2 * rbf_kernel(first, second, bandwidth).mean()
    )
    return max(0.0, float(value)), float(bandwidth)


def mmd_permutation_test(
    first,
    second,
    permutations=DEFAULT_PERMUTATIONS,
    seed=42,
    permutation_limit=200,
    paired=False,
    bandwidth=None,
):
    """Test equality of feature distributions with a fixed-bandwidth MMD.

    Paired image views are permuted by swapping the domain label within each
    image pair. Unpaired samples use the ordinary pooled-label permutation.
    """
    first, second = _as_matrix(first), _as_matrix(second)
    if paired and len(first) != len(second):
        raise ValueError("Paired MMD requires the same number of rows per domain")
    full_mmd, full_bandwidth = maximum_mean_discrepancy(first, second, bandwidth)
    generator = np.random.default_rng(seed)
    if paired and len(first) > permutation_limit:
        indices = generator.choice(len(first), permutation_limit, replace=False)
        first, second = first[indices], second[indices]
    elif not paired:
        if len(first) > permutation_limit:
            first = first[
                generator.choice(len(first), permutation_limit, replace=False)
            ]
        if len(second) > permutation_limit:
            second = second[
                generator.choice(len(second), permutation_limit, replace=False)
            ]

    combined = np.concatenate((first, second), axis=0)
    first_size = len(first)
    second_size = len(second)
    kernel = rbf_kernel(combined, combined, full_bandwidth)

    observed_weights = np.concatenate(
        (
            np.full(first_size, 1 / first_size),
            np.full(second_size, -1 / second_size),
        )
    )
    observed = max(0.0, float(observed_weights @ kernel @ observed_weights))
    exceedances = 0
    completed = 0
    while completed < permutations:
        batch_size = min(PERMUTATION_BATCH_SIZE, permutations - completed)
        if paired:
            pair_signs = generator.choice(
                (-1.0, 1.0), size=(batch_size, first_size)
            )
            weights = np.concatenate((pair_signs, -pair_signs), axis=1)
            weights /= first_size
        else:
            random_order = generator.random((batch_size, len(combined)))
            first_indices = np.argpartition(
                random_order, first_size - 1, axis=1
            )[:, :first_size]
            weights = np.full(
                (batch_size, len(combined)), -1 / second_size, dtype=np.float64
            )
            np.put_along_axis(
                weights,
                first_indices,
                np.full_like(first_indices, 1 / first_size, dtype=np.float64),
                axis=1,
            )
        permuted = np.einsum(
            "bi,ij,bj->b", weights, kernel, weights, optimize=True
        )
        exceedances += int(np.count_nonzero(permuted >= observed))
        completed += batch_size
    p_value = (exceedances + 1) / (permutations + 1)
    return {
        "mmd_squared": full_mmd,
        "mmd_bandwidth": full_bandwidth,
        "mmd_permutation_p": float(p_value),
        "mmd_permutations": int(permutations),
        "mmd_permutation_resolution": float(1 / (permutations + 1)),
        "mmd_permutation_samples_per_group": int(min(permutation_limit, len(first))),
        "mmd_permutation_scheme": "paired_within_image_swap"
        if paired
        else "pooled_label_permutation",
    }


def domain_classifier_auc(first, second, seed=42, folds=5, paired=False):
    """Estimate how easily a linear classifier separates two domains."""
    first, second = _as_matrix(first), _as_matrix(second)
    if paired and len(first) != len(second):
        raise ValueError(
            "Paired domain classification requires the same number of rows per domain"
        )
    features = np.concatenate((first, second), axis=0)
    labels = np.concatenate((np.zeros(len(first)), np.ones(len(second))))
    fold_count = min(folds, len(first), len(second))
    if fold_count < 2:
        raise ValueError("Domain classification requires at least two folds")
    groups = None
    if paired:
        groups = np.concatenate((np.arange(len(first)), np.arange(len(second))))
        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
    else:
        splitter = StratifiedKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
    )
    probabilities = cross_val_predict(
        classifier,
        features,
        labels,
        cv=splitter,
        groups=groups,
        method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(labels, probabilities))


def distribution_shift_metrics(
    first,
    second,
    *,
    permutations=DEFAULT_PERMUTATIONS,
    permutation_limit=200,
    seed=42,
    paired=False,
    bandwidth=None,
):
    metrics = mmd_permutation_test(
        first,
        second,
        permutations,
        seed,
        permutation_limit,
        paired,
        bandwidth,
    )
    metrics["domain_classifier_roc_auc"] = domain_classifier_auc(
        first, second, seed, paired=paired
    )
    metrics["domain_classifier_cv"] = (
        "pair_grouped_stratified" if paired else "stratified"
    )
    return metrics
