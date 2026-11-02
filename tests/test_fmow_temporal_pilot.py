import unittest

import numpy as np

from fyp_drift.evaluation.fmow_temporal_pilot import (
    fdr_bh,
    matched_metric_sample,
    score_predictions,
)


class FMoWTemporalPilotTests(unittest.TestCase):
    def test_metric_sample_matches_class_country_composition(self):
        rows_by_period = {}
        features_by_period = {}
        labels_by_period = {}
        for period, offset in (("history", 0), ("future", 100)):
            rows = []
            labels = []
            for class_name, country, count in (
                ("airport", "USA", 6),
                ("airport", "CAN", 4),
                ("port", "USA", 2),
            ):
                for item in range(count):
                    rows.append({
                        "period": period,
                        "category": class_name,
                        "country": country,
                        "local_path": f"{period}-{class_name}-{country}-{item}",
                    })
                    labels.append(0 if class_name == "airport" else 1)
            rows_by_period[period] = rows
            labels_by_period[period] = np.asarray(labels)
            features_by_period[period] = np.arange(offset, offset + len(rows))[:, None]

        features, labels = matched_metric_sample(
            rows_by_period, features_by_period, labels_by_period, 6, 7
        )
        self.assertEqual(len(features["history"]), 6)
        self.assertEqual(len(features["future"]), 6)
        self.assertEqual(
            np.bincount(labels["history"], minlength=2).tolist(),
            np.bincount(labels["future"], minlength=2).tolist(),
        )
        self.assertTrue(np.all(features["history"][:, 0] < 100))
        self.assertTrue(np.all(features["future"][:, 0] >= 100))

    def test_benjamini_hochberg_adjustment_is_monotonic_by_rank(self):
        adjusted = fdr_bh([0.04, 0.001, 0.02])
        self.assertAlmostEqual(adjusted[1], 0.003)
        self.assertLessEqual(adjusted[1], adjusted[2])
        self.assertLessEqual(adjusted[2], adjusted[0])

    def test_task_scores_include_accuracy_macro_f1_and_balanced_accuracy(self):
        result = score_predictions(np.asarray([0, 0, 1, 1]), np.asarray([
            [2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 2.0],
        ]))
        self.assertEqual(result["accuracy"], 0.75)
        self.assertIn("macro_f1", result)
        self.assertIn("balanced_accuracy", result)


if __name__ == "__main__":
    unittest.main()
