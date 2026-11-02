import json
import tempfile
import unittest
from pathlib import Path

from fyp_drift.training.fine_tune import parse_args, validate_checkpoint


class FineTuneProtocolTests(unittest.TestCase):
    def test_defaults_match_the_long_horizon_comparison(self):
        args = parse_args(["--base-checkpoint", "historical.pt"])
        self.assertEqual(args.adapt_year, 2010)
        self.assertEqual(args.calibration_year, 2011)
        self.assertEqual(args.test_years, (2012, 2013, 2014))
        self.assertEqual(args.drift_scenarios, ("clean", "realistic_mix"))
        self.assertEqual(args.selection_fraction, 0.5)
        self.assertFalse(args.reuse_calibration_for_selection)

    def test_selection_fraction_is_configurable_and_bounded(self):
        args = parse_args(
            ["--base-checkpoint", "historical.pt", "--selection-fraction", "0.4"]
        )
        self.assertEqual(args.selection_fraction, 0.4)
        with self.assertRaises(SystemExit):
            parse_args(
                ["--base-checkpoint", "historical.pt", "--selection-fraction", "1"]
            )

    def test_checkpoint_must_be_pre_adaptation_uniform_baseline(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            run = Path(temporary_directory)
            checkpoint = run / "models" / "best.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            (run / "config.json").write_text(
                json.dumps({"method": "baseline", "cv_val_year": 2009}),
                encoding="utf-8",
            )
            validate_checkpoint(checkpoint)

            (run / "config.json").write_text(
                json.dumps({"method": "xai", "cv_val_year": 2009}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "baseline checkpoint"):
                validate_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
