"""DINOv2 ViT classifier with an optional saliency gate inside the forward pass."""
import torch
from torch import nn
from transformers import Dinov2Model

BACKBONE = "facebook/dinov2-small"


class TemporalViT(nn.Module):
    def __init__(self, method="baseline"):
        super().__init__()
        if method not in ("baseline", "xai"):
            raise ValueError(method)
        self.method = method
        self.backbone = Dinov2Model.from_pretrained(BACKBONE)
        width = self.backbone.config.hidden_size
        self.saliency_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.classifier = nn.Linear(width * 2, 62)

    def forward(self, image, return_counterfactual=False):
        tokens = self.backbone(pixel_values=image).last_hidden_state
        cls, patches = tokens[:, 0], tokens[:, 1:]
        n = patches.size(1)
        if self.method == "xai":
            gate = self.saliency_head(patches).squeeze(-1).sigmoid()
            saliency = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            saliency = patches.new_full((patches.size(0), n), 1 / n)
        selected = torch.sum(patches * saliency.unsqueeze(-1), dim=1)
        logits = self.classifier(torch.cat((cls, selected), dim=1))
        counterfactual = None
        if return_counterfactual and self.method == "xai":
            inverse_gate = 1 - gate
            inverse = inverse_gate / inverse_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            removed = torch.sum(patches * inverse.unsqueeze(-1), dim=1)
            counterfactual = self.classifier(torch.cat((cls, removed), dim=1))
        return logits, saliency, counterfactual
