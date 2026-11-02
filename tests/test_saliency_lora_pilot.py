import unittest

import torch
from torch import nn

from fyp_drift.training.saliency_lora_pilot import (
    LoRALinear, consistency_weights, teacher_saliency,
)


class SaliencyLoRAPilotTests(unittest.TestCase):
    def test_zero_initialized_lora_preserves_historical_projection(self):
        base = nn.Linear(3, 2)
        lora = LoRALinear(base, rank=2)
        inputs = torch.randn(4, 5, 3)
        self.assertTrue(torch.equal(base(inputs), lora(inputs)))
        self.assertFalse(any(p.requires_grad for p in base.parameters()))
        self.assertTrue(all(p.requires_grad for p in lora.up.parameters()))

    def test_class_evidence_selects_top_patches_and_normalizes(self):
        tokens = torch.tensor([[[4.0], [1.0], [3.0], [0.0], [2.0]]])
        weights = torch.tensor([[1.0], [-1.0]])
        saliency = teacher_saliency(tokens, weights, torch.tensor([0]), 0.4)
        self.assertEqual(saliency.tolist(), [[0.5, 0.0, 0.5, 0.0, 0.0]])
        self.assertTrue(torch.equal(saliency.sum(1), torch.ones(1)))
        self.assertTrue(torch.equal(
            consistency_weights("uniform", saliency), torch.full_like(saliency, 0.2)
        ))


if __name__ == "__main__":
    unittest.main()
