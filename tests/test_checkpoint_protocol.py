import json
import tempfile
import unittest
from pathlib import Path

from fyp_drift.training.temporal_router import validate_historical_checkpoint


class HistoricalCheckpointProtocolTests(unittest.TestCase):
    def test_long_horizon_accepts_only_pre_adaptation_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            run = Path(temporary_directory)
            checkpoint = run / "models" / "best.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            (run / "config.json").write_text(
                json.dumps({"method": "xai", "cv_val_year": 2009}),
                encoding="utf-8",
            )
            validate_historical_checkpoint(checkpoint, long_horizon=True)

    def test_long_horizon_rejects_checkpoint_trained_through_2010(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            run = Path(temporary_directory)
            checkpoint = run / "models" / "best.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            (run / "config.json").write_text(
                json.dumps({"method": "xai", "cv_val_year": None}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cv-val-year 2009"):
                validate_historical_checkpoint(checkpoint, long_horizon=True)


if __name__ == "__main__":
    unittest.main()
