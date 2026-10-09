import unittest

import numpy as np
import torch
from PIL import Image

from fyp_drift.data.corruptions import (
    REALISTIC_BENCHMARK_SCENARIOS,
    drift_transform,
    realistic_mixed_drift_transform,
    scenario_components,
)


class RealisticDriftTest(unittest.TestCase):
    def setUp(self):
        horizontal = np.linspace(20, 235, 224, dtype=np.uint8)[None, :, None]
        vertical = np.linspace(235, 20, 224, dtype=np.uint8)[:, None, None]
        image = np.concatenate(
            (
                np.broadcast_to(horizontal, (224, 224, 1)),
                np.broadcast_to(vertical, (224, 224, 1)),
                np.full((224, 224, 1), 128, dtype=np.uint8),
            ),
            axis=2,
        )
        self.image = Image.fromarray(image, "RGB")

    def test_every_benchmark_source_is_finite_and_deterministic(self):
        for scenario in REALISTIC_BENCHMARK_SCENARIOS:
            with self.subTest(scenario=scenario):
                transform = drift_transform(scenario, severity=3, seed=42)
                first = transform(self.image, 123)
                second = transform(self.image, 123)
                self.assertEqual(first.shape, (3, 224, 224))
                self.assertTrue(torch.isfinite(first).all())
                self.assertTrue(torch.equal(first, second))

    def test_realistic_mix_varies_by_sample_and_occlusion_is_minor(self):
        compositions = [
            scenario_components("realistic_mix", index, 3, 42) for index in range(1000)
        ]
        self.assertGreater(len(set(compositions)), 20)
        occluded = sum("occlusion" in composition for composition in compositions)
        self.assertGreaterEqual(occluded, 70)
        self.assertLessEqual(occluded, 130)
        self.assertTrue(all(3 <= len(composition) <= 4 for composition in compositions))

    def test_adaptation_stream_contains_clean_and_each_individual_source(self):
        transform = realistic_mixed_drift_transform((1, 2, 3), seed=42)
        outputs = [transform(self.image, index) for index in range(30)]
        self.assertTrue(all(output.shape == (3, 224, 224) for output in outputs))
        self.assertGreater(len({output.numpy().tobytes() for output in outputs}), 10)


if __name__ == "__main__":
    unittest.main()
