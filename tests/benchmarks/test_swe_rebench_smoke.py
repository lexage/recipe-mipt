import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import smoke_swe_rebench


class SmokeCLITests(unittest.TestCase):
    def test_smoke_runs_inference_then_single_instance_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = [
                "smoke_swe_rebench.py",
                "--config",
                "pipeline.yaml",
                "--instance-id",
                "owner__repo-1",
                "--fork-path",
                "/fork",
                "--output-dir",
                temporary,
                "--model-name-or-path",
                "react-sgr/model",
                "--run-id",
                "smoke-1",
                "--evaluation-check-only",
            ]
            successful = type("Result", (), {"returncode": 0})()
            with (
                patch.object(sys, "argv", arguments),
                patch(
                    "smoke_swe_rebench.subprocess.run",
                    side_effect=(successful, successful),
                ) as run,
            ):
                self.assertEqual(smoke_swe_rebench.main(), 0)

            inference, evaluation = [call.args[0] for call in run.call_args_list]
            self.assertIn("run_swe_rebench.py", inference[1])
            self.assertIn("evaluate_swe_rebench.py", evaluation[1])
            self.assertIn("--check-only", evaluation)
            report_index = evaluation.index("--report-dir")
            self.assertEqual(
                evaluation[report_index + 1], str(Path(temporary) / "evaluation")
            )
            self.assertEqual(
                (Path(temporary) / "instance_ids.txt").read_text(),
                "owner__repo-1\n",
            )

    def test_smoke_stops_when_inference_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = [
                "smoke_swe_rebench.py",
                "--config",
                "pipeline.yaml",
                "--instance-id",
                "owner__repo-1",
                "--fork-path",
                "/fork",
                "--output-dir",
                temporary,
                "--model-name-or-path",
                "model",
                "--run-id",
                "smoke-1",
            ]
            failed = type("Result", (), {"returncode": 7})()
            with (
                patch.object(sys, "argv", arguments),
                patch("smoke_swe_rebench.subprocess.run", return_value=failed) as run,
            ):
                self.assertEqual(smoke_swe_rebench.main(), 7)
                run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
