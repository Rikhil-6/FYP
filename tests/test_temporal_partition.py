import unittest
from typing import ClassVar

from fyp_drift.data.clear100 import (
    ClearSubset,
    balanced_indices,
    stratified_year_partition_loaders,
    year_loader,
)
from fyp_drift.evaluation.drift_verification import stratified_indices


class _Dataset:
    metadata_fields: ClassVar = ["year"]

    def __init__(self, rows):
        self.rows = rows

    def get_subset(self, split, transform=None):
        return ClearSubset(".", self.rows, transform)


class TemporalPartitionTests(unittest.TestCase):
    def test_limited_sample_represents_classes_and_years(self):
        rows = [
            {
                "path": f"unused-{year}-{label}-{item}.jpg",
                "label": str(label),
                "year": str(year),
            }
            for year in (2008, 2009)
            for label in range(4)
            for item in range(5)
        ]
        selected = balanced_indices(rows, range(len(rows)), 16, seed=11)
        self.assertEqual({int(rows[index]["year"]) for index in selected}, {2008, 2009})
        self.assertEqual({int(rows[index]["label"]) for index in selected}, set(range(4)))

    def test_selection_and_calibration_are_disjoint_stratified_and_deterministic(self):
        rows = [
            {"path": f"unused-{label}-{item}.jpg", "label": str(label), "year": "2011"}
            for label in range(100)
            for item in range(7)
        ]
        dataset = _Dataset(rows)
        first = stratified_year_partition_loaders(
            dataset, "train_source", 2011, seed=17
        )
        second = stratified_year_partition_loaders(
            dataset, "train_source", 2011, seed=17
        )
        selection = first[0].dataset.indices
        calibration = first[1].dataset.indices

        self.assertEqual(selection, second[0].dataset.indices)
        self.assertEqual(calibration, second[1].dataset.indices)
        self.assertFalse(set(selection) & set(calibration))
        self.assertEqual(set(selection) | set(calibration), set(range(len(rows))))
        for label in range(100):
            selected = sum(int(rows[index]["label"]) == label for index in selection)
            calibrated = sum(
                int(rows[index]["label"]) == label for index in calibration
            )
            self.assertLessEqual(abs(selected - calibrated), 1)

    def test_missing_requested_year_is_rejected(self):
        rows = [
            {"path": f"unused-{label}.jpg", "label": str(label), "year": "2011"}
            for label in range(100)
        ]
        with self.assertRaisesRegex(ValueError, "2012"):
            year_loader(_Dataset(rows), "train_source", (2011, 2012))

    def test_drift_sample_cannot_silently_drop_classes(self):
        rows = [
            {"path": f"unused-{label}.jpg", "label": str(label), "year": "2014"}
            for label in range(100)
        ]
        subset = ClearSubset(".", rows)
        with self.assertRaisesRegex(ValueError, "cannot represent all"):
            stratified_indices(subset, 2014, limit=99, seed=42)


if __name__ == "__main__":
    unittest.main()
