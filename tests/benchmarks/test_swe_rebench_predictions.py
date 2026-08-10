import json
import tempfile
import threading
import unittest
from pathlib import Path

from src.benchmarks.swe_rebench import (
    PredictionsWriter,
    RunArtifactsWriter,
    SWERebenchDataError,
    SWERebenchPrediction,
)


class PredictionsWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "predictions.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_writes_strict_evaluator_record(self):
        writer = PredictionsWriter(self.path, model_name_or_path="react/model")
        writer.write("owner__repo-1", "diff --git a/a.py b/a.py\n")
        record = json.loads(self.path.read_text())
        self.assertEqual(
            set(record), {"instance_id", "model_name_or_path", "model_patch"}
        )
        self.assertEqual(record["model_name_or_path"], "react/model")

    def test_resume_and_duplicate_protection(self):
        writer = PredictionsWriter(self.path, model_name_or_path="model")
        writer.write("owner__repo-1", "patch")
        resumed = PredictionsWriter(self.path, model_name_or_path="model", resume=True)
        self.assertEqual(resumed.completed_ids, {"owner__repo-1"})
        with self.assertRaisesRegex(SWERebenchDataError, "already exists"):
            resumed.write("owner__repo-1", "other")
        with self.assertRaisesRegex(SWERebenchDataError, "use resume"):
            PredictionsWriter(self.path, model_name_or_path="model")

    def test_parallel_writes_remain_valid_jsonl(self):
        writer = PredictionsWriter(self.path, model_name_or_path="model")
        threads = [
            threading.Thread(target=writer.write, args=(f"instance-{index}", "patch"))
            for index in range(20)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len(records), 20)
        self.assertEqual(len({record["instance_id"] for record in records}), 20)

    def test_rejects_invalid_existing_record_and_extra_fields(self):
        self.path.write_text('{"instance_id": "id", "extra": true}\n')
        with self.assertRaisesRegex(SWERebenchDataError, "line 1"):
            PredictionsWriter(self.path, model_name_or_path="model", resume=True)
        with self.assertRaises(SWERebenchDataError):
            SWERebenchPrediction("", "model", "patch")

    def test_operational_artifacts_are_separate(self):
        artifacts = RunArtifactsWriter(self.root)
        artifacts.write_metadata({"run_id": "run"})
        artifacts.write_error("instance", "inference", RuntimeError("failed"))
        self.assertEqual(
            json.loads((self.root / "run_metadata.json").read_text())["run_id"], "run"
        )
        error = json.loads((self.root / "errors.jsonl").read_text())
        self.assertEqual(error["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
