import tempfile
import unittest
from pathlib import Path

from efficient_sampling import cli


REPO_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_discover_repo_root(self):
        self.assertEqual(cli.discover_repo_root(str(REPO_ROOT)), REPO_ROOT)

    def test_stage_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "ackley-fmin"
            staged = cli.stage_benchmark(REPO_ROOT, "tableII", "ackley", "fmin", output)
            self.assertEqual(staged, output)
            self.assertTrue((output / "main_workflow.py").exists())
            self.assertTrue((output / "_model.py").exists())

    def test_stage_eos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "Skyrme"
            staged = cli.stage_eos(REPO_ROOT, "Skyrme", output)
            self.assertEqual(staged, output)
            self.assertTrue((output / "main_workflow.py").exists())

    def test_stage_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "ocp"
            staged = cli.stage_md(REPO_ROOT, "ocp", output)
            self.assertEqual(staged, output)
            self.assertTrue((output / "main_workflow.py").exists())


if __name__ == "__main__":
    unittest.main()
