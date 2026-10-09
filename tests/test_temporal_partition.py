import unittest
from typing import ClassVar

from fyp_drift.data.clear100 import ClearSubset, stratified_year_partition_loaders


class _Dataset:
    metadata_fields: ClassVar = ["year"]

    def __init__(self, rows):
        self.rows = rows

    def get_subset(self, split, transform=None):
        return ClearSubset(".", self.rows, transform)


class TemporalPartitionTests(unittest.TestCase):
    def test_selection_and_calibration_are_disjoint_stratified_and_deterministic(self):
        rows = [
            {"path": f"unused-{label}-{item}.jpg", "label": str(label), "year": "2011"}
            for label in range(4)
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
        for label in range(4):
            selected = sum(int(rows[index]["label"]) == label for index in selection)
            calibrated = sum(
                int(rows[index]["label"]) == label for index in calibration
            )
            self.assertLessEqual(abs(selected - calibrated), 1)


if __name__ == "__main__":
    unittest.main()
