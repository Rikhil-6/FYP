import json
import tempfile
import unittest
from pathlib import Path

import torch

from fyp_drift.artifacts.run_store import RunStore


class RunStoreCheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_resume_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = RunStore(root, "siamese", "round-trip")
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

            loss = model(torch.ones(1, 3)).sum()
            loss.backward()
            optimizer.step()
            expected = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }

            store.save_training_checkpoint(model, optimizer, epoch=2, best_score=0.75)
            payload = torch.load(
                store.path / "models" / "resume.pt", weights_only=False
            )
            metadata = json.loads((store.path / "models" / "resume.json").read_text())

            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(payload["best_score"], 0.75)
            self.assertIn("state", payload["optimizer"])
            self.assertEqual(metadata["epoch"], 2)
            self.assertEqual(metadata["best_score"], 0.75)
            self.assertIn("saved_utc", metadata)
            for name, value in expected.items():
                self.assertTrue(torch.equal(payload["model"][name], value))


if __name__ == "__main__":
    unittest.main()
