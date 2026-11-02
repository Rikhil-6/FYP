import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fyp_drift.evaluation.adaptation_evidence import (
    compare_checkpoints,
    compare_feature_archives,
)


class AdaptationEvidenceTests(unittest.TestCase):
    def test_aligned_features_reduce_measured_condition_shift(self):
        generator = np.random.default_rng(12)
        normal = generator.normal(size=(80, 12))
        historical = {
            "normal_features": normal,
            "adverse_features": normal
            + 1.2
            + generator.normal(0, 0.15, normal.shape),
        }
        adapted = {
            "normal_features": normal,
            "adverse_features": normal + generator.normal(0, 0.08, normal.shape),
        }

        report = compare_feature_archives(
            historical,
            adapted,
            permutations=39,
            bootstrap_repetitions=100,
            seed=12,
        )

        self.assertGreater(report["change"]["mmd_squared_reduction"], 0)
        self.assertGreater(report["change"]["domain_auc_reduction"], 0)
        self.assertEqual(
            report["historical"]["distribution_shift"]["mmd_bandwidth"],
            report["adapted"]["distribution_shift"]["mmd_bandwidth"],
        )
        paired = report["change"]["paired_feature_distance"]
        self.assertGreater(paired["historical_minus_adapted_mean"], 0)
        self.assertGreater(paired["bootstrap_95_ci"][0], 0)
        self.assertTrue(report["conclusions"]["baseline_drift"]["detected"])
        self.assertTrue(
            report["conclusions"]["representation_mitigation"]["supported"]
        )
        self.assertEqual(
            report["conclusions"]["overall"],
            "representation_mitigation_supported_task_not_assessed",
        )

    def test_task_evidence_requires_adverse_gain_and_clean_retention(self):
        generator = np.random.default_rng(4)
        normal = generator.normal(size=(40, 6))
        labels = np.arange(40) % 2

        def probabilities(predictions):
            values = np.full((40, 2), 0.1)
            values[np.arange(40), predictions] = 0.9
            return values

        historical_predictions = labels.copy()
        historical_predictions[:12] = 1 - historical_predictions[:12]
        adapted_predictions = labels.copy()
        adapted_predictions[:4] = 1 - adapted_predictions[:4]
        shared = {
            "normal_features": normal,
            "normal_labels": labels,
            "adverse_labels": labels,
            "normal_probabilities": probabilities(labels),
        }
        historical = {
            **shared,
            "adverse_features": normal + 1,
            "adverse_probabilities": probabilities(historical_predictions),
        }
        adapted = {
            **shared,
            "adverse_features": normal + generator.normal(0, 0.05, normal.shape),
            "adverse_probabilities": probabilities(adapted_predictions),
        }

        report = compare_feature_archives(
            historical,
            adapted,
            permutations=19,
            bootstrap_repetitions=50,
            seed=4,
        )

        self.assertEqual(
            report["conclusions"]["task_mitigation"]["status"], "supported"
        )
        self.assertGreater(
            report["conclusions"]["task_mitigation"][
                "adverse_macro_f1_change"
            ],
            0,
        )

    def test_checkpoint_comparison_reports_parameter_movement(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            historical = torch.nn.Linear(4, 2)
            adapted = torch.nn.Linear(4, 2)
            adapted.load_state_dict(historical.state_dict())
            with torch.no_grad():
                adapted.weight.add_(0.1)
            first = root / "historical.pt"
            second = root / "adapted.pt"
            torch.save(historical.state_dict(), first)
            torch.save(adapted.state_dict(), second)

            report = compare_checkpoints(first, second)

            self.assertGreater(report["relative_l2_change"], 0)
            self.assertGreater(report["changed_parameter_fraction"], 0)
            self.assertEqual(
                report["largest_relative_changes"][0]["parameter"], "weight"
            )


if __name__ == "__main__":
    unittest.main()
