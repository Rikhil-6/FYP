"""Twin ViT comparison and class-conditional, saliency-only routing."""

import torch
from torch import nn

from fyp_drift.models.vision_transformer import TemporalViT


def divergence(first, second):
    """Jensen-Shannon divergence of the two learned patch saliency maps."""
    midpoint = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.clamp_min(1e-8) / midpoint.clamp_min(1e-8)).log()).sum(1)
        + (second * (second.clamp_min(1e-8) / midpoint.clamp_min(1e-8)).log()).sum(1)
    )


class ViTTwinRouter(nn.Module):
    """Compare frozen historical and drift-adapted XAI ViT snapshots."""

    def __init__(self, historical: TemporalViT, updated: TemporalViT):
        super().__init__()
        if historical.method != "xai" or updated.method != "xai":
            raise ValueError("Both towers must use the saliency-gated ViT")
        self.historical = historical
        self.updated = updated

    @property
    def old(self):
        """Compatibility alias used by archived analysis code."""
        return self.historical

    @property
    def new(self):
        """Compatibility alias used by archived checkpoints and reports."""
        return self.updated

    def compare(self, images):
        with torch.no_grad():
            old_logits, old_map, old_inverse = self.historical(
                images, return_counterfactual=True
            )
        new_logits, new_map, new_inverse = self.updated(
            images, return_counterfactual=True
        )
        saliency_gap = divergence(old_map, new_map)
        old_class = old_logits.argmax(1)
        new_class = new_logits.argmax(1)
        old_evidence = (
            (old_logits.softmax(1) - old_inverse.softmax(1))
            .gather(1, old_class[:, None])
            .squeeze(1)
        )
        new_evidence = (
            (new_logits.softmax(1) - new_inverse.softmax(1))
            .gather(1, new_class[:, None])
            .squeeze(1)
        )
        return {
            "old_logits": old_logits,
            "new_logits": new_logits,
            "new_inverse": new_inverse,
            "old_map": old_map,
            "new_map": new_map,
            "saliency_gap": saliency_gap,
            "old_evidence": old_evidence,
            "new_evidence": new_evidence,
            "score": saliency_gap,
        }

    @staticmethod
    def route(comparison, thresholds):
        old_logits, new_logits = comparison["old_logits"], comparison["new_logits"]
        old_class = old_logits.argmax(1)
        new_class = new_logits.argmax(1)
        boundary = torch.tensor(
            [thresholds.get(int(c), thresholds["global"]) for c in old_class],
            device=old_logits.device,
        )
        needs_update = (comparison["score"] > boundary) & (old_class != new_class)
        use_new = needs_update & (
            comparison["new_evidence"] > comparison["old_evidence"]
        )
        prediction = torch.where(use_new, new_class, old_class)
        return prediction, needs_update, use_new


def fit_thresholds(class_scores, quantile=0.9, minimum=5):
    """Calibrate on correctly classified old-model examples from the calibration window."""
    values = [score for scores in class_scores.values() for score in scores]
    if not values:
        raise ValueError("No correct calibration examples were available")
    global_threshold = float(torch.tensor(values).quantile(quantile))
    thresholds = {"global": global_threshold}
    for class_id, scores in class_scores.items():
        if len(scores) >= minimum:
            thresholds[class_id] = float(torch.tensor(scores).quantile(quantile))
    return thresholds
