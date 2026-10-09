"""DINOv2 ViT classifier with an optional saliency gate inside the forward pass."""

import torch
from torch import nn
from transformers import Dinov2Model

BACKBONE = "facebook/dinov2-small"


class TemporalViT(nn.Module):
    """Classify images from the class token and a weighted patch summary.

    The baseline uses equal patch weights. The XAI variant learns one positive
    weight per patch, so its saliency map directly changes the classifier input.
    """

    def __init__(self, method="baseline", num_classes=100):
        super().__init__()
        if method not in ("baseline", "xai"):
            raise ValueError(method)
        self.method = method
        self.backbone = Dinov2Model.from_pretrained(BACKBONE)
        width = self.backbone.config.hidden_size
        self.saliency_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, images, return_counterfactual=False):
        tokens = self.backbone(pixel_values=images).last_hidden_state
        class_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        patch_count = patch_tokens.size(1)
        if self.method == "xai":
            patch_gates = self.saliency_head(patch_tokens).squeeze(-1).sigmoid()
            saliency = patch_gates / patch_gates.sum(dim=-1, keepdim=True).clamp_min(
                1e-8
            )
        else:
            saliency = patch_tokens.new_full(
                (patch_tokens.size(0), patch_count), 1 / patch_count
            )
        selected_patches = torch.sum(patch_tokens * saliency.unsqueeze(-1), dim=1)
        logits = self.classifier(torch.cat((class_token, selected_patches), dim=1))
        counterfactual = None
        if return_counterfactual and self.method == "xai":
            inverse_gates = 1 - patch_gates
            inverse_saliency = inverse_gates / inverse_gates.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            complementary_patches = torch.sum(
                patch_tokens * inverse_saliency.unsqueeze(-1), dim=1
            )
            counterfactual = self.classifier(
                torch.cat((class_token, complementary_patches), dim=1)
            )
        return logits, saliency, counterfactual
