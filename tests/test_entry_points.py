import importlib
import tomllib
import unittest
from pathlib import Path


class EntryPointTests(unittest.TestCase):
    def test_every_console_command_resolves_to_a_callable(self):
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "pyproject.toml").open("rb") as handle:
            scripts = tomllib.load(handle)["project"]["scripts"]

        self.assertGreaterEqual(len(scripts), 10)
        for command, target in scripts.items():
            with self.subTest(command=command):
                module_name, function_name = target.split(":", maxsplit=1)
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, function_name)))


if __name__ == "__main__":
    unittest.main()
