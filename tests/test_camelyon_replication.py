import unittest

from fyp_drift.evaluation.camelyon_replication import (
    balanced_rows,
    balanced_source_rows,
)


class FakeStream:
    def __init__(self, rows):
        self.rows = rows

    def shuffle(self, **_):
        return self.rows


class FakeImage:
    def convert(self, _):
        return self

    def copy(self):
        return self


class CamelyonReplicationTests(unittest.TestCase):
    def test_balanced_rows_returns_equal_classes(self):
        rows = [
            {"image": FakeImage(), "label": label}
            for label in (0, 0, 0, 1, 1, 1)
        ]
        selected = balanced_rows(FakeStream(rows), limit=4, seed=42)
        self.assertEqual([label for _, label in selected], [0, 1, 0, 1])

    def test_balanced_rows_rejects_odd_limit(self):
        with self.assertRaises(ValueError):
            balanced_rows(FakeStream([]), limit=3, seed=42)

    def test_source_splits_are_balanced(self):
        rows = [
            {
                "image": FakeImage(),
                "label": label,
                "center": centre,
                "patient": patient,
            }
            for patient in range(100)
            for centre in (0, 3, 4)
            for label in (0, 1)
        ]
        selected = balanced_source_rows(
            FakeStream(rows), train_limit=20, eval_limit=10, seed=42
        )
        self.assertEqual(sum(label for _, label in selected["train"]), 10)
        self.assertEqual(sum(label for _, label in selected["id"]), 5)


if __name__ == "__main__":
    unittest.main()
