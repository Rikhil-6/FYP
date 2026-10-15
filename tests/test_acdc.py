import tempfile
import unittest
from pathlib import Path

from fyp_drift.data.acdc import scan_acdc, write_manifest


class AcdcManifestTests(unittest.TestCase):
    def test_scanner_matches_adverse_reference_label_and_invalid_mask(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for condition in ("fog", "night", "rain", "snow"):
                recording = f"GOPR-{condition}"
                stem = f"{recording}_frame_000001"
                paths = (
                    root / "rgb_anon" / condition / "train" / recording / f"{stem}_rgb_anon.png",
                    root / "rgb_anon" / condition / "train_ref" / recording / f"{stem}_rgb_ref_anon.png",
                    root / "gt" / condition / "train" / recording / f"{stem}_gt_labelTrainIds.png",
                    root / "gt" / condition / "train" / recording / f"{stem}_gt_invIds.png",
                )
                for path in paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

            rows = scan_acdc(root, splits=("train",))
            manifest = write_manifest(rows, root / "paired_manifest.csv")

            self.assertEqual(len(rows), 4)
            self.assertTrue(manifest.is_file())
            self.assertTrue(all(row["normal_image"] for row in rows))
            self.assertTrue(all(row["invalid_mask"] for row in rows))


if __name__ == "__main__":
    unittest.main()
