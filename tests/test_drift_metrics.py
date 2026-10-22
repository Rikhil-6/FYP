import unittest

import numpy as np

from fyp_drift.evaluation.drift_metrics import (
    distribution_shift_metrics,
    maximum_mean_discrepancy,
    mmd_permutation_test,
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
            paired=True,
        )
        self.assertGreater(metrics["mmd_squared"], 0.05)
        self.assertLessEqual(metrics["mmd_permutation_p"], 0.05)
        self.assertGreater(metrics["domain_classifier_roc_auc"], 0.9)
        self.assertEqual(
            metrics["mmd_permutation_scheme"], "paired_within_image_swap"
        )
        self.assertEqual(metrics["mmd_permutation_resolution"], 1 / 40)
        self.assertEqual(metrics["domain_classifier_cv"], "pair_grouped_stratified")

    def test_paired_test_rejects_unequal_pair_counts(self):
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            mmd_permutation_test(
                self.reference,
                self.shifted[:-1],
                permutations=19,
                paired=True,
            )

    def test_fixed_bandwidth_is_reported_and_reused(self):
        metrics = mmd_permutation_test(
            self.reference,
            self.shifted,
            permutations=19,
            paired=True,
            bandwidth=0.75,
        )
        self.assertEqual(metrics["mmd_bandwidth"], 0.75)


if __name__ == "__main__":
    unittest.main()
