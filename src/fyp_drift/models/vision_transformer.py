"""DINOv2 ViT classifier with an optional saliency gate inside the forward pass."""

import torch
from torch import nn
from transformers import Dinov2Model

BACKBONE = "facebook/dinov2-small"


class TemporalViT(nn.Module):
    """Classify images from a patch summary with an optional direct CLS path.

    The baseline uses equal patch weights; XAI learns one positive weight per
    patch. ``patch_only`` removes the direct class-token-to-head bypass while
    retaining the backbone's class/patch attention interactions.
    """

    def __init__(self, method="baseline", num_classes=100,
                 pooling_mode="class_plus_patch"):
        super().__init__()
        if method not in ("baseline", "xai"):
            raise ValueError(method)
        if pooling_mode not in ("class_plus_patch", "patch_only"):
            raise ValueError(pooling_mode)
        self.method = method
        self.pooling_mode = pooling_mode
        self.backbone = Dinov2Model.from_pretrained(BACKBONE)
        width = self.backbone.config.hidden_size
        self.saliency_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.classifier = nn.Linear(
            width * (2 if pooling_mode == "class_plus_patch" else 1), num_classes
        )

    def encode(self, images):
        """Return the classified representation and its patch weights."""
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
            patch_gates = patch_tokens.new_ones((patch_tokens.size(0), patch_count))
            saliency = patch_tokens.new_full(
                (patch_tokens.size(0), patch_count), 1 / patch_count
            )
        selected_patches = torch.sum(patch_tokens * saliency.unsqueeze(-1), dim=1)
        representation = (
            torch.cat((class_token, selected_patches), dim=1)
            if self.pooling_mode == "class_plus_patch"
            else selected_patches
        )
        return representation, saliency, patch_tokens, patch_gates

    def forward(self, images, return_counterfactual=False):
        representation, saliency, patch_tokens, patch_gates = self.encode(images)
        logits = self.classifier(representation)
        counterfactual = None
        if return_counterfactual and self.method == "xai":
            inverse_gates = 1 - patch_gates
            inverse_saliency = inverse_gates / inverse_gates.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            complementary_patches = torch.sum(
                patch_tokens * inverse_saliency.unsqueeze(-1), dim=1
            )
            complement_representation = (
                torch.cat(
                    (representation[:, : patch_tokens.size(-1)], complementary_patches),
                    dim=1,
                )
                if self.pooling_mode == "class_plus_patch"
                else complementary_patches
            )
            counterfactual = self.classifier(complement_representation)
        return logits, saliency, counterfactual

    def patch_weight_interventions(self, images, permutation=None):
        """Score the same tokens with learned and controlled patch weights.

        This is an inference diagnostic. It holds the backbone and classifier
        fixed, so any learned-versus-uniform difference comes from patch
        weighting. Class-only and patch-only scores reveal bypass capacity but
        are out-of-training-distribution interventions, not trained controls.
        """
        if self.method != "xai":
            raise ValueError("Patch-weight interventions require an XAI model")
        representation, saliency, patch_tokens, patch_gates = self.encode(images)
        patch_count = patch_tokens.size(1)
        if permutation is None:
            permutation = torch.randperm(patch_count, device=patch_tokens.device)
        if permutation.ndim != 1 or permutation.numel() != patch_count:
            raise ValueError("Permutation must have one index per patch")
        if not torch.equal(
            torch.sort(permutation.detach().cpu()).values,
            torch.arange(patch_count),
        ):
            raise ValueError("Permutation must contain every patch index once")
        permutation = permutation.to(device=patch_tokens.device)
        width = patch_tokens.size(-1)
        class_token = representation[:, :width] if self.pooling_mode == "class_plus_patch" else None
        learned_patches = (
            representation[:, width:]
            if self.pooling_mode == "class_plus_patch"
            else representation
        )
        uniform_patches = patch_tokens.mean(dim=1)
        permuted_patches = torch.sum(
            patch_tokens * saliency[:, permutation].unsqueeze(-1), dim=1
        )
        inverse_gates = 1 - patch_gates
        inverse_saliency = inverse_gates / inverse_gates.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        inverted_patches = torch.sum(
            patch_tokens * inverse_saliency.unsqueeze(-1), dim=1
        )

        def classify(patches):
            data = (
                torch.cat((class_token, patches), dim=1)
                if class_token is not None
                else patches
            )
            return self.classifier(data)

        results = {
            "learned": classify(learned_patches),
            "uniform": classify(uniform_patches),
            "permuted": classify(permuted_patches),
            "inverted": classify(inverted_patches),
        }
        if class_token is not None:
            results["class_only"] = self.classifier(
                torch.cat((class_token, torch.zeros_like(learned_patches)), dim=1)
            )
            results["patch_only"] = self.classifier(
                torch.cat((torch.zeros_like(class_token), learned_patches), dim=1)
            )
        return results
