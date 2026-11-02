import unittest

from fyp_drift.evaluation.wilds_replication import (
    stratified_group_sample,
    stratified_sample,
)


class WildsReplicationTests(unittest.TestCase):
    def test_stratified_sample_represents_each_class(self):
        labels = [0] * 20 + [1] * 4 + [2] * 2
        indices = stratified_sample(labels, limit=9, seed=7)
        selected_labels = {labels[index] for index in indices}
        self.assertEqual(selected_labels, {0, 1, 2})
        self.assertEqual(len(indices), 9)
        self.assertEqual(len(indices), len(set(indices)))

    def test_limit_smaller_than_class_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot represent all"):
            stratified_sample([0, 1, 2], limit=2, seed=7)

    def test_group_sample_represents_each_year_and_class(self):
        labels = [0, 1] * 8
        years = [2000] * 4 + [2001] * 4 + [2002] * 4 + [2003] * 4
        indices = stratified_group_sample(labels, years, limit=8, seed=7)
        represented = {(years[index], labels[index]) for index in indices}
        self.assertEqual(len(represented), 8)


if __name__ == "__main__":
    unittest.main()
