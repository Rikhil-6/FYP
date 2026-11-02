import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from fyp_drift.models.vision_transformer import TemporalViT
from fyp_drift.training.temporal_router import build_router


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1)
        self.encoder = SimpleNamespace(layer=nn.ModuleList([nn.Linear(1, 1)] * 3))

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=pixel_values)


class PatchMechanismTests(unittest.TestCase):
    def make_model(self):
        with patch(
            "fyp_drift.models.vision_transformer.Dinov2Model.from_pretrained",
            return_value=TinyBackbone(),
        ):
            model = TemporalViT("xai", num_classes=2)
        model.saliency_head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.saliency_head.weight.fill_(1)
        return model

    def test_intervention_exposes_class_token_bypass(self):
        model = self.make_model()
        tokens = torch.tensor([[[2.0], [1.0], [3.0]]])
        permutation = torch.tensor([1, 0])

        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()
            model.classifier.weight[0, 0] = 1  # Class-token-only head.
            cls = model.patch_weight_interventions(tokens, permutation)
        self.assertTrue(torch.equal(cls["learned"], cls["uniform"]))
        self.assertTrue(torch.equal(cls["learned"], cls["permuted"]))

        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.weight[0, 1] = 1  # Patch-only head.
            patches = model.patch_weight_interventions(tokens, permutation)
            _, _, complement = model(tokens, return_counterfactual=True)
        self.assertNotEqual(float(patches["learned"][0, 0]),
                            float(patches["uniform"][0, 0]))
        self.assertNotEqual(float(patches["learned"][0, 0]),
                            float(patches["permuted"][0, 0]))
        self.assertLess(float(patches["inverted"][0, 0]),
                        float(patches["uniform"][0, 0]))
        inverse = 1 - torch.sigmoid(torch.tensor([1.0, 3.0]))
        expected_patch = (inverse * torch.tensor([1.0, 3.0])).sum() / inverse.sum()
        self.assertTrue(torch.allclose(
            patches["inverted"], model.classifier(
                torch.tensor([[2.0, expected_patch]])
            ), atol=1e-6
        ))
        self.assertTrue(torch.equal(patches["inverted"], complement))
        self.assertTrue(torch.equal(patches["learned"], patches["patch_only"]))

    def test_rejects_repeated_patch_index(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "every patch index"):
            model.patch_weight_interventions(
                torch.tensor([[[2.0], [1.0], [3.0]]]), torch.tensor([0, 0])
            )

    def test_patch_only_head_has_no_direct_class_token_path(self):
        with patch(
            "fyp_drift.models.vision_transformer.Dinov2Model.from_pretrained",
            return_value=TinyBackbone(),
        ):
            model = TemporalViT("xai", num_classes=2, pooling_mode="patch_only")
        model.saliency_head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.saliency_head.weight.fill_(1)
            model.classifier.weight.fill_(1)
            model.classifier.bias.zero_()
            first = torch.tensor([[[2.0], [1.0], [3.0]]])
            changed_class_token = torch.tensor([[[20.0], [1.0], [3.0]]])
            first_logits, _, first_complement = model(
                first, return_counterfactual=True
            )
            second_logits, _, second_complement = model(
                changed_class_token, return_counterfactual=True
            )
            interventions = model.patch_weight_interventions(
                first, torch.tensor([1, 0])
            )
        self.assertTrue(torch.equal(first_logits, second_logits))
        self.assertTrue(torch.equal(first_complement, second_complement))
        self.assertNotIn("class_only", interventions)

    def test_shared_uniform_checkpoint_starts_with_same_predictions(self):
        with patch(
            "fyp_drift.models.vision_transformer.Dinov2Model.from_pretrained",
            side_effect=lambda *_: TinyBackbone(),
        ):
            baseline = TemporalViT(
                "baseline", num_classes=2, pooling_mode="patch_only"
            )
            with tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                (run / "models").mkdir()
                checkpoint = run / "models" / "best.pt"
                torch.save(baseline.state_dict(), checkpoint)
                (run / "config.json").write_text(json.dumps({
                    "method": "baseline", "cv_val_year": 2009,
                    "pooling_mode": "patch_only",
                }))
                router, _, _, _ = build_router(
                    checkpoint, "cpu", "realistic",
                    initialize_saliency_from_uniform=True,
                )
                tokens = torch.tensor([[[2.0], [1.0], [3.0]]])
                with torch.no_grad():
                    baseline_logits = baseline(tokens)[0]
                    old_logits, old_map, _ = router.old(tokens)
                self.assertTrue(torch.allclose(old_logits, baseline_logits))
                self.assertTrue(torch.allclose(
                    old_map, torch.full_like(old_map, 0.5)
                ))


if __name__ == "__main__":
    unittest.main()
