import json
import sys
import tempfile
import unittest
from pathlib import Path

from src.benchmarks.swe_rebench import (
    EvaluationConfig,
    SWERebenchDataError,
    SWERebenchEvaluator,
    validate_predictions,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.predictions = self.root / "predictions.jsonl"
        self.predictions.write_text(
            json.dumps(
                {
                    "instance_id": "owner__repo-1",
                    "model_name_or_path": "react-sgr/model",
                    "model_patch": "diff --git a/a b/a\n",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_prediction_validation_is_strict(self):
        records = validate_predictions(
            self.predictions, allowed_instance_ids={"owner__repo-1"}
        )
        self.assertEqual(records[0].instance_id, "owner__repo-1")

        invalid = self.root / "invalid.jsonl"
        invalid.write_text(
            '{"instance_id":"x","model_name_or_path":"m","model_patch":"","extra":1}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SWERebenchDataError, "exactly"):
            validate_predictions(invalid)

    def test_prediction_validation_rejects_duplicates_and_unknown_ids(self):
        self.predictions.write_text(
            self.predictions.read_text(encoding="utf-8") * 2, encoding="utf-8"
        )
        with self.assertRaisesRegex(SWERebenchDataError, "Duplicate"):
            validate_predictions(self.predictions)

        self.predictions.write_text(
            self.predictions.read_text(encoding="utf-8").splitlines()[0] + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SWERebenchDataError, "selected dataset"):
            validate_predictions(self.predictions, allowed_instance_ids={"another"})

    def test_compatibility_gate_and_command(self):
        self._write_fake_harness()
        config = self._config(instance_ids=("owner__repo-1",))
        evaluator = SWERebenchEvaluator(config, python=sys.executable)

        help_text = evaluator.check_compatibility()
        self.assertIn("--predictions_path", help_text)
        command = evaluator.command()
        self.assertEqual(command[-2:], ["--instance_ids", "owner__repo-1"])
        self.assertIn(str(self.predictions.resolve()), command)
        self.assertIsNone(evaluator.run(check_only=True))
        metadata = json.loads(
            (self.root / "reports" / "evaluation_metadata.json").read_text()
        )
        self.assertEqual(metadata["fork_commit"], None)
        self.assertEqual(metadata["run_id"], "smoke")
        self.assertEqual(metadata["command"], command)

    def test_compatibility_gate_rejects_wrong_fork(self):
        package = self.root / "swebench" / "harness"
        package.mkdir(parents=True)
        (self.root / "swebench" / "__init__.py").touch()
        (package / "__init__.py").touch()
        (package / "run_evaluation.py").write_text(
            "import argparse\nargparse.ArgumentParser().parse_args()\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "missing options"):
            SWERebenchEvaluator(
                self._config(), python=sys.executable
            ).check_compatibility()

    def test_evaluation_requires_complete_ids_and_matching_metadata(self):
        self._write_fake_harness()
        config = self._config(instance_ids=("owner__repo-1", "owner__repo-2"))
        with self.assertRaisesRegex(SWERebenchDataError, "missing"):
            SWERebenchEvaluator(config, python=sys.executable).run(check_only=True)

        metadata = self.root / "run_metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "dataset": "nebius/SWE-rebench",
                    "dataset_revision": "revision-1",
                    "split": "test",
                    "selected_instance_ids": ["owner__repo-1"],
                }
            ),
            encoding="utf-8",
        )
        config = self._config(
            instance_ids=("owner__repo-1",),
            inference_metadata=metadata,
            dataset_revision="wrong-revision",
        )
        with self.assertRaisesRegex(SWERebenchDataError, "dataset_revision"):
            SWERebenchEvaluator(config, python=sys.executable).run(check_only=True)

    def _config(
        self, *, instance_ids=(), inference_metadata=None, dataset_revision=None
    ):
        return EvaluationConfig(
            fork_path=self.root,
            predictions_path=self.predictions,
            dataset_name="nebius/SWE-rebench",
            dataset_revision=dataset_revision,
            split="test",
            run_id="smoke",
            instance_ids=instance_ids,
            report_dir=self.root / "reports",
            inference_metadata=inference_metadata,
        )

    def _write_fake_harness(self):
        package = self.root / "swebench" / "harness"
        package.mkdir(parents=True)
        (self.root / "swebench" / "__init__.py").touch()
        (package / "__init__.py").touch()
        (package / "run_evaluation.py").write_text(
            """import argparse
p = argparse.ArgumentParser()
p.add_argument('--dataset_name')
p.add_argument('--split')
p.add_argument('--predictions_path')
p.add_argument('--max_workers')
p.add_argument('--timeout')
p.add_argument('--run_id')
p.add_argument('--namespace')
p.add_argument('--instance_image_tag')
p.add_argument('--report_dir')
p.add_argument('--instance_ids', nargs='*')
p.parse_args()
""",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
