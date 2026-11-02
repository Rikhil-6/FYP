import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fyp_drift.evaluation.router_feature_ablation import (
    _fixed_coverage_route,
    _native_route,
    _policy_reproduction,
    _source_config,
    _summary,
)
from fyp_drift.evaluation.metrics import classification_metrics


class RouterFeatureAblationTest(unittest.TestCase):
    def test_source_must_use_corrected_disjoint_split(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config_path = run_dir / "config.json"
            base = {"long_horizon": True, "reuse_calibration_for_selection": False}
            for protocol in ("shared_legacy", None):
                config_path.write_text(
                    json.dumps(base | {"calibration_protocol": protocol}),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    _source_config(run_dir)
            config_path.write_text(
                json.dumps(base | {"calibration_protocol": "disjoint_class_stratified"}),
                encoding="utf-8",
            )
            self.assertTrue(_source_config(run_dir)["long_horizon"])

    def test_native_floor_and_fixed_coverage_answer_different_questions(self):
        features = np.array(
            [
                [0.05, 0.9, 0, 0],
                [0.80, 0.8, 0, 0],
                [0.90, 0.7, 0, 0],
                [0.95, 0.6, 0, 0],
                [0.01, 0.5, 0, 0],
            ],
            dtype=np.float32,
        )
        old = np.zeros(5, dtype=np.int64)
        new = np.ones(5, dtype=np.int64)
        policy = {
            "feature_names": ["confidence_gain"],
            "mean": [0],
            "scale": [1],
            "coefficients": [1],
            "intercept": 0,
            "saliency_floor": 0.5,
            "probability_threshold": 0.5,
        }
        probabilities, native = _native_route(features, old, new, policy)
        self.assertEqual(native.tolist(), [False, True, True, True, False])
        matched = _fixed_coverage_route(probabilities, old, new, 0.4)
        self.assertEqual(matched.tolist(), [True, True, False, False, False])

    def test_same_route_quota_and_macro_f1_matches_project_metric(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        old = np.array([0, 1, 1, 0, 2, 1])
        new = np.array([1, 0, 0, 1, 1, 2])
        probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        route = _fixed_coverage_route(probabilities, old, new, 0.5)
        self.assertEqual(int(route.sum()), 3)
        result = _summary(truth, old, new, route, num_classes=3)
        prediction = np.where(route, new, old)
        expected = classification_metrics(truth, prediction, classes=3)
        self.assertAlmostEqual(result["routed"]["macro_f1"], expected["macro_f1"], places=4)
        self.assertEqual(result["routed_to_new"], 3)

    def test_refit_check_detects_calibration_route_change(self):
        features = np.array([[0, -1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        old = np.zeros(2, dtype=np.int64)
        new = np.ones(2, dtype=np.int64)
        source = {
            "feature_names": ["confidence_gain"],
            "mean": [0], "scale": [1], "coefficients": [1],
            "intercept": 0, "saliency_floor": None,
            "probability_threshold": 0.5,
        }
        refitted = source | {"probability_threshold": 0.9}
        check = _policy_reproduction(features, old, new, source, refitted)
        self.assertFalse(check["threshold_match"])
        self.assertEqual(check["calibration_route_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
