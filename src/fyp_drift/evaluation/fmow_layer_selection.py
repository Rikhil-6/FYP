"""Choose DINOv2 blocks using frozen-head patch evidence and target images."""

from __future__ import annotations

import numpy as np

from fyp_drift.evaluation.drift_metrics import maximum_mean_discrepancy
from fyp_drift.training.saliency_lora_pilot import LoRALinear


GROUPS = {"E": range(0, 4), "M": range(4, 8), "L": range(8, 12)}


def evidence_weights(scores: np.ndarray, mode: str, seed: int, keep_fraction: float = 0.4) -> np.ndarray:
    """Equal-area saliency and shuffled controls; rows are images, columns patches."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or not 0 < keep_fraction <= 1:
        raise ValueError("scores must be N x P and keep_fraction must be in (0, 1]")
    n, patches = scores.shape
    if mode == "uniform":
        return np.full((n, patches), 1 / patches, dtype=np.float32)
    if mode not in ("saliency", "shuffled"):
        raise ValueError(mode)
    count = max(1, round(keep_fraction * patches))
    indices = np.argpartition(-scores, count - 1, axis=1)[:, :count]
    weights = np.zeros((n, patches), dtype=np.float32)
    np.put_along_axis(weights, indices, 1 / count, axis=1)
    if mode == "shuffled":
        rng = np.random.default_rng(seed)
        weights = np.take_along_axis(weights, np.argsort(rng.random(weights.shape), axis=1), axis=1)
    return weights


def layer_features(tokens: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool each block's patch tokens with one fixed map per image."""
    tokens = np.asarray(tokens, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if tokens.ndim != 4 or tokens.shape[0] != len(weights) or tokens.shape[2] != weights.shape[1]:
        raise ValueError("expected tokens N x 12 x P x D and weights N x P")
    return np.einsum("np,nlpd->nld", weights, tokens, optimize=True)


def profile(source_tokens, target_tokens, source_scores, target_scores, seed=42) -> dict:
    """MMD profile and locked E/M/L pick for each weighting mode."""
    result = {}
    for mode in ("saliency", "uniform", "shuffled"):
        source = layer_features(source_tokens, evidence_weights(source_scores, mode, seed))
        target = layer_features(target_tokens, evidence_weights(target_scores, mode, seed + 1))
        layer_mmd = [maximum_mean_discrepancy(source[:, i], target[:, i])[0]
                     for i in range(source.shape[1])]
        if len(layer_mmd) != 12:
            raise ValueError("expected twelve DINOv2 blocks")
        group_scores = {name: float(np.mean([layer_mmd[i] for i in indices]))
                        for name, indices in GROUPS.items()}
        pick = max(GROUPS, key=lambda name: group_scores[name])
        result[mode] = {"layer_mmd_squared": layer_mmd, "group_scores": group_scores, "pick": pick}
    return result


def add_group_lora(backbone, group: str, rank: int = 8, alpha: int = 16) -> int:
    """Freeze DINOv2 and train query/key/value/projection/MLP LoRA plus block norms."""
    if group not in (*GROUPS, "ALL"):
        raise ValueError(group)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    indices = range(12) if group == "ALL" else GROUPS[group]
    blocks = backbone.encoder.layer
    if len(blocks) != 12:
        raise ValueError("expected a 12-block DINOv2 model")
    for index in indices:
        block = blocks[index]
        attention = block.attention.attention
        for name in ("query", "key", "value"):
            setattr(attention, name, LoRALinear(getattr(attention, name), rank, alpha))
        block.attention.output.dense = LoRALinear(block.attention.output.dense, rank, alpha)
        block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank, alpha)
        block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank, alpha)
        for name in ("norm1", "norm2"):
            for parameter in getattr(block, name).parameters():
                parameter.requires_grad = True
    return sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad)
