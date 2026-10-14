import unittest

import numpy as np

from fyp_drift.evaluation.drift_metrics import (
    distribution_shift_metrics,
    maximum_mean_discrepancy,
)


class DriftMetricTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(7)
        self.reference = generator.normal(size=(80, 8))
        self.shifted = self.reference + 1.25

    def test_identical_features_have_zero_distances(self):
        mmd, _ = maximum_mean_discrepancy(self.reference, self.reference)
        self.assertAlmostEqual(mmd, 0.0, places=10)

    def test_shifted_features_are_detectable(self):
        metrics = distribution_shift_metrics(
            self.reference,
            self.shifted,
            permutations=39,
            seed=7,
        )
        self.assertGreater(metrics["mmd_squared"], 0.05)
        self.assertLessEqual(metrics["mmd_permutation_p"], 0.05)
        self.assertGreater(metrics["domain_classifier_roc_auc"], 0.9)


if __name__ == "__main__":
    unittest.main()
