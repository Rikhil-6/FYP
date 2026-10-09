"""Interpretable calibration policy for deciding when the newer ViT should be used."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from fyp_drift.evaluation.metrics import classification_metrics

FEATURES = ("saliency_js", "confidence_gain", "entropy_reduction", "evidence_gain")


def feature_tensor(comparison):
    old_p = comparison["old_logits"].softmax(1)
    new_p = comparison["new_logits"].softmax(1)
    old_conf = old_p.max(1).values
    new_conf = new_p.max(1).values
    old_entropy = -(old_p * old_p.clamp_min(1e-8).log()).sum(1)
    new_entropy = -(new_p * new_p.clamp_min(1e-8).log()).sum(1)
    return torch.stack(
        (
            comparison["saliency_gap"],
            new_conf - old_conf,
            old_entropy - new_entropy,
            comparison["new_evidence"] - comparison["old_evidence"],
        ),
        1,
    )


def fit_policy(features, labels, old_predictions, new_predictions, truth):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    mean, scale = features.mean(0), features.std(0).clip(1e-8)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit((features - mean) / scale, labels)
    probabilities = model.predict_proba((features - mean) / scale)[:, 1]
    best = None
    for quantile in (0.20, 0.35, 0.50, 0.65):
        saliency_floor = float(np.quantile(features[:, 0], quantile))
        for threshold in np.linspace(0.1, 0.9, 33):
            use_new = (
                (old_predictions != new_predictions)
                & (features[:, 0] >= saliency_floor)
                & (probabilities >= threshold)
            )
            routed = np.where(use_new, new_predictions, old_predictions)
            score = classification_metrics(truth, routed)["macro_f1"]
            harmful = int(
                np.sum(
                    use_new & (old_predictions == truth) & (new_predictions != truth)
                )
            )
            candidate = (score, -harmful, -quantile, float(threshold), saliency_floor)
            if best is None or candidate > best:
                best = candidate
    return {
        "feature_names": list(FEATURES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "probability_threshold": best[3],
        "saliency_floor": best[4],
        "saliency_quantile": -best[2],
        "calibration_macro_f1": best[0],
        "calibration_harmful_routes": -best[1],
    }


def apply_policy(comparison, policy):
    features = feature_tensor(comparison)
    mean = features.new_tensor(policy["mean"])
    scale = features.new_tensor(policy["scale"])
    weights = features.new_tensor(policy["coefficients"])
    probability = torch.sigmoid(
        ((features - mean) / scale * weights).sum(1) + policy["intercept"]
    )
    old_class = comparison["old_logits"].argmax(1)
    new_class = comparison["new_logits"].argmax(1)
    saliency_gate = features[:, 0] >= policy["saliency_floor"]
    needs_update = (old_class != new_class) & saliency_gate
    use_new = needs_update & (probability >= policy["probability_threshold"])
    return (
        torch.where(use_new, new_class, old_class),
        needs_update,
        use_new,
        probability,
        features,
    )
