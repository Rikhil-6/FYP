import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from fyp_drift.evaluation.fmow_temporal_pilot import (
    apply_fmow_corruption,
    clustered_bootstrap_delta,
    fdr_bh,
    matched_metric_sample,
    matched_period_sample,
    save_corruption_preview,
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

    def test_gaussian_stress_uses_deterministic_increasing_levels(self):
        image = Image.new("RGB", (32, 32), (127, 127, 127))
        outputs = [
            apply_fmow_corruption(image, "gaussian_noise", level, 17, 4)
            for level in (1, 2, 3)
        ]
        repeated = apply_fmow_corruption(image, "gaussian_noise", 2, 17, 4)
        self.assertTrue(np.array_equal(outputs[1].numpy(), repeated.numpy()))
        clean = apply_fmow_corruption(image, "clean", 1, 17, 4)
        errors = [float(((value - clean) ** 2).mean()) for value in outputs]
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])

    def test_task_scoring_accepts_precomputed_tta_predictions(self):
        result = score_predictions(np.asarray([0, 1, 1]), np.asarray([0, 0, 1]))
        self.assertAlmostEqual(result["accuracy"], 2 / 3)

    def test_cross_period_stress_sample_matches_every_period(self):
        splits = {}
        for period in ("historical_2010_13", "adapt_2014_15", "future_2016_17"):
            splits[period] = [
                {"period": period, "category": "airport", "country": "USA", "local_path": f"{period}-{i}"}
                for i in range(5)
            ] + [
                {"period": period, "category": "port", "country": "CAN", "local_path": f"{period}-p-{i}"}
                for i in range(3)
            ]
        selected = matched_period_sample(splits, 4, 12)
        self.assertEqual({period: len(rows) for period, rows in selected.items()}, {
            "historical_2010_13": 4, "adapt_2014_15": 4, "future_2016_17": 4,
        })
        compositions = [
            sorted((row["category"], row["country"]) for row in selected[period])
            for period in selected
        ]
        self.assertEqual(compositions[0], compositions[1])
        self.assertEqual(compositions[1], compositions[2])

    def test_clustered_bootstrap_returns_paired_score_interval(self):
        result = clustered_bootstrap_delta(
            [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1],
            ["a", "a", "b", "b"], repetitions=20, seed=3,
        )
        self.assertEqual(result["accuracy"]["candidate_minus_reference"], 0.5)
        self.assertEqual(len(result["accuracy"]["cluster_bootstrap_95_ci"]), 2)

    def test_corruption_preview_is_saved_with_pillow(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.jpg"
            output = Path(directory) / "preview.png"
            Image.new("RGB", (32, 32), (120, 90, 70)).save(source)
            save_corruption_preview(
                [{
                    "category": "airport", "local_path": str(source),
                    "period": "future_2016_17", "timestamp": "2017-01-01",
                    "country": "USA", "utm": "10T",
                }], output
            )
            self.assertTrue(output.is_file())
            examples = Path(directory) / "examples"
            self.assertTrue((examples / "manifest.csv").is_file())
            saved_images = list(examples.glob("*.jpg")) + list(examples.glob("*.png"))
            self.assertEqual(len(saved_images), 6)
            with Image.open(output) as preview:
                self.assertEqual(preview.size[1], 224 + 28 + 34)


if __name__ == "__main__":
    unittest.main()
