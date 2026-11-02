import unittest

import numpy as np

from fyp_drift.models.routing import FEATURES, FEATURE_SETS, fit_policy


class RoutingFeatureAblationTest(unittest.TestCase):
    def setUp(self):
        self.features = np.array(
            [
                [0.01, -0.20, -0.10, -0.15],
                [0.02, -0.10, -0.05, -0.08],
                [0.03, 0.05, 0.03, 0.02],
                [0.04, 0.10, 0.08, 0.06],
                [0.08, 0.20, 0.15, 0.12],
                [0.09, 0.25, 0.18, 0.16],
                [0.11, 0.30, 0.22, 0.20],
                [0.12, 0.35, 0.28, 0.25],
            ]
        )
        self.labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.old = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.new = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        self.truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def fit(self, feature_set):
        return fit_policy(
            self.features,
            self.labels,
            self.old,
            self.new,
            self.truth,
            feature_names=FEATURE_SETS[feature_set],
        )

    def test_feature_sets_are_explicit_and_recorded(self):
        for feature_set, expected in FEATURE_SETS.items():
            with self.subTest(feature_set=feature_set):
                policy = self.fit(feature_set)
                self.assertEqual(policy["feature_names"], list(expected))
                self.assertEqual(len(policy["coefficients"]), len(expected))

    def test_prediction_only_policy_has_no_saliency_floor(self):
        policy = self.fit("confidence_only")
        self.assertIsNone(policy["saliency_floor"])
        self.assertIsNone(policy["saliency_quantile"])

    def test_saliency_policies_retain_the_floor(self):
        for feature_set in ("saliency_only", "combined"):
            with self.subTest(feature_set=feature_set):
                policy = self.fit(feature_set)
                self.assertIsNotNone(policy["saliency_floor"])
                self.assertIn("saliency_js", policy["feature_names"])

    def test_declared_features_match_model_order(self):
        self.assertEqual(FEATURE_SETS["combined"], FEATURES)

    def test_no_beneficial_switches_produces_safe_constant_policy(self):
        policy = fit_policy(
            self.features,
            np.zeros(len(self.labels), dtype=int),
            self.old,
            self.new,
            self.truth,
        )
        self.assertEqual(policy["fit_kind"], "constant")
        self.assertEqual(policy["beneficial_switch_rate"], 0.0)
        self.assertTrue(all(value == 0.0 for value in policy["coefficients"]))


if __name__ == "__main__":
    unittest.main()
