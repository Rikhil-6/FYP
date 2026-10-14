import unittest

from fyp_drift.evaluation.wilds_replication import stratified_sample


class WildsReplicationTests(unittest.TestCase):
    def test_stratified_sample_represents_each_class(self):
        labels = [0] * 20 + [1] * 4 + [2] * 2
        indices = stratified_sample(labels, limit=9, seed=7)
        selected_labels = {labels[index] for index in indices}
        self.assertEqual(selected_labels, {0, 1, 2})
        self.assertEqual(len(indices), 9)
        self.assertEqual(len(indices), len(set(indices)))


if __name__ == "__main__":
    unittest.main()
