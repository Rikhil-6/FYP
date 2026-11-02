import unittest

import numpy as np
import torch

from transformers import Dinov2Config, Dinov2Model

from fyp_drift.evaluation.fmow_layer_selection import (
    add_group_lora, evidence_weights, layer_features, profile,
)


class LayerSelectionTests(unittest.TestCase):
    def test_equal_area_controls(self):
        scores = np.random.default_rng(3).normal(size=(8, 20))
        saliency = evidence_weights(scores, "saliency", 42)
        shuffled = evidence_weights(scores, "shuffled", 42)
        np.testing.assert_allclose(saliency.sum(1), 1)
        np.testing.assert_allclose(shuffled.sum(1), 1)
        np.testing.assert_array_equal((saliency > 0).sum(1), np.full(8, 8))
        np.testing.assert_array_equal((shuffled > 0).sum(1), np.full(8, 8))
        np.testing.assert_allclose(evidence_weights(scores, "uniform", 42).sum(1), 1)

    def test_profile_uses_salient_patch_shift_to_pick_group(self):
        rng = np.random.default_rng(5)
        source = rng.normal(size=(24, 12, 8, 2)).astype(np.float32)
        target = source.copy()
        target[:, :4, :3] += 3.0
        scores = np.tile(np.array([5, 5, 5, 0, 0, 0, 0, 0]), (24, 1))
        result = profile(source, target, scores, scores)
        self.assertEqual(result["saliency"]["pick"], "E")
        self.assertEqual(len(result["saliency"]["layer_mmd_squared"]), 12)

    def test_layer_features_shape(self):
        tokens = np.ones((2, 12, 4, 3))
        pooled = layer_features(tokens, np.full((2, 4), 0.25))
        self.assertEqual(pooled.shape, (2, 12, 3))
        np.testing.assert_allclose(pooled, 1)

    def test_lora_groups_have_equal_parameter_counts_and_forward(self):
        config = Dinov2Config(image_size=28, patch_size=14, hidden_size=24,
                              num_hidden_layers=12, num_attention_heads=4,
                              intermediate_size=48)
        counts = []
        for group in ("E", "M", "L"):
            model = Dinov2Model(config)
            counts.append(add_group_lora(model, group))
            output = model(pixel_values=torch.randn(1, 3, 28, 28))
            self.assertEqual(output.last_hidden_state.shape, (1, 5, 24))
        self.assertEqual(len(set(counts)), 1)


if __name__ == "__main__":
    unittest.main()
