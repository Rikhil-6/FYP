import unittest
from collections import Counter

from fyp_drift.data.temporal_sampling import matched_temporal_sample
from fyp_drift.evaluation.temporal_design import (
    crossed_corruption_grid,
    progressive_corruption_schedule,
)


class TemporalDesignTests(unittest.TestCase):
    def test_crossed_grid_applies_each_severity_in_every_period(self):
        cells = crossed_corruption_grid(
            ("history", "adapt", "future"),
            ("gaussian_noise", "sensor_noise"),
        )
        self.assertEqual(len(cells), 21)
        for period in ("history", "adapt", "future"):
            period_cells = [cell for cell in cells if cell["period"] == period]
            self.assertEqual(
                Counter(cell["scenario"] for cell in period_cells),
                Counter({"clean": 1, "gaussian_noise": 3, "sensor_noise": 3}),
            )
            self.assertEqual(
                [cell["severity"] for cell in period_cells if cell["scenario"] == "gaussian_noise"],
                [1, 2, 3],
            )

    def test_crossed_grid_rejects_a_mix_whose_source_count_changes_with_severity(self):
        with self.assertRaisesRegex(ValueError, "changes source count"):
            crossed_corruption_grid(("history", "future"), ("realistic_mix",))

    def test_progression_is_explicit_and_monotonic(self):
        cells = progressive_corruption_schedule(
            ("history", "adapt", "future"), "gaussian_noise", (0, 1, 2)
        )
        self.assertEqual(
            [(cell["scenario"], cell["severity"]) for cell in cells],
            [("clean", 0), ("gaussian_noise", 1), ("gaussian_noise", 2)],
        )
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            progressive_corruption_schedule(
                ("history", "adapt", "future"), "gaussian_noise", (0, 2, 1)
            )

    def test_matched_sample_has_identical_class_country_counts(self):
        rows = []
        for period in ("history", "adapt", "future"):
            for category, country, count in (
                ("crop", "USA", 5),
                ("crop", "CAN", 3),
                ("wind", "USA", 2),
            ):
                for item in range(count):
                    rows.append(
                        {
                            "period": period,
                            "category": category,
                            "country": country,
                            "sequence": f"{period}-{category}-{country}-{item}",
                        }
                    )
        selected = matched_temporal_sample(
            rows,
            period_key="period",
            strata_keys=("category", "country"),
            target_per_period=6,
            seed=9,
            group_key="sequence",
        )
        counts = Counter(
            (row["_matched_period"], row["category"], row["country"])
            for row in selected
        )
        for stratum in (("crop", "USA"), ("crop", "CAN"), ("wind", "USA")):
            observed = [counts[(period, *stratum)] for period in ("history", "adapt", "future")]
            self.assertEqual(len(set(observed)), 1)
        self.assertEqual(
            Counter(row["_matched_period"] for row in selected),
            Counter({"history": 6, "adapt": 6, "future": 6}),
        )

    def test_matched_sampler_rejects_insufficient_common_support(self):
        rows = [
            {"period": "history", "class": "a", "country": "US"},
            {"period": "future", "class": "b", "country": "US"},
        ]
        with self.assertRaisesRegex(ValueError, "matched capacity 0"):
            matched_temporal_sample(
                rows,
                period_key="period",
                strata_keys=("class", "country"),
                target_per_period=1,
            )

    def test_matched_sampler_rejects_sequence_leakage_across_periods(self):
        rows = [
            {"period": "history", "class": "a", "country": "US", "sequence": "same"},
            {"period": "adapt", "class": "a", "country": "US", "sequence": "same"},
            {"period": "future", "class": "a", "country": "US", "sequence": "future"},
        ]
        with self.assertRaisesRegex(ValueError, "appears in periods"):
            matched_temporal_sample(
                rows,
                period_key="period",
                strata_keys=("class", "country"),
                target_per_period=1,
                group_key="sequence",
            )


if __name__ == "__main__":
    unittest.main()
