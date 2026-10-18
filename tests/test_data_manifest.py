import copy
import unittest

from fyp_drift.data.manifest import expected_split, validate_manifest_rows


def complete_manifest():
    rows = []
    for source, archive_root in (("train", "train_image_only"), ("test", "test")):
        for bucket in range(11):
            for label in range(100):
                class_name = f"class-{label:03d}"
                rows.append(
                    {
                        "path": (
                            f"{archive_root}/labeled_images/{bucket}/{class_name}/"
                            f"{source}-{bucket}-{label}.jpg"
                        ),
                        "source": source,
                        "bucket": str(bucket),
                        "year": str(2004 + bucket),
                        "class_name": class_name,
                        "label": str(label),
                        "split": expected_split(source, bucket),
                    }
                )
    return rows


class ClearManifestTests(unittest.TestCase):
    def setUp(self):
        self.rows = complete_manifest()

    def test_complete_temporal_manifest_is_valid(self):
        validate_manifest_rows(self.rows)

    def test_missing_class_in_one_bucket_is_rejected(self):
        incomplete = [
            row
            for row in self.rows
            if not (
                row["source"] == "test"
                and row["bucket"] == "8"
                and row["label"] == "99"
            )
        ]
        with self.assertRaisesRegex(ValueError, "missing classes"):
            validate_manifest_rows(incomplete)

    def test_metadata_that_disagrees_with_path_is_rejected(self):
        inconsistent = copy.deepcopy(self.rows)
        inconsistent[0]["year"] = "2005"
        with self.assertRaisesRegex(ValueError, "year/bucket mismatch"):
            validate_manifest_rows(inconsistent)

    def test_cross_period_image_reuse_is_rejected(self):
        duplicated = copy.deepcopy(self.rows)
        duplicated[100]["path"] = duplicated[100]["path"].replace(
            "train-1-0.jpg", "train-0-0.jpg"
        )
        with self.assertRaisesRegex(ValueError, "reuses an image identifier"):
            validate_manifest_rows(duplicated)


if __name__ == "__main__":
    unittest.main()
