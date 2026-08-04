import json
import tempfile
import threading
import unittest
from pathlib import Path

from src.benchmarks.swe_rebench import (
    CommandResult,
    InstanceImageResolver,
    PredictionsWriter,
    RepositoryRuntime,
    RunArtifactsWriter,
    SWERebenchInferenceRunner,
    SWERebenchTask,
    get_repository_runtime,
)


class FakePipeline:
    def __init__(self, fail_id=None):
        self.fail_id = fail_id
        self.closed = False
        self.runtime_seen = None

    def run(self, prompt):
        self.runtime_seen = get_repository_runtime().instance_id
        if self.runtime_seen == self.fail_id:
            raise RuntimeError("pipeline failed")
        return "ignored final answer"

    def close(self):
        self.closed = True


class FakeRuntime(RepositoryRuntime):
    def __init__(self, task, patch="diff --git a/a.py b/a.py\n"):
        super().__init__(task.instance_id, task.base_commit, "/testbed")
        self.patch = patch

    def list_files(self, path=".", **kwargs):
        return ""

    def read_file(self, path, **kwargs):
        return ""

    def search_code(self, query, **kwargs):
        return ""

    def apply_patch(self, patch, **kwargs):
        return ""

    def run_command(self, command, **kwargs):
        return CommandResult(command, 0, "", "", 0.0)

    def get_diff(self, path=".", **kwargs):
        return ""

    def get_patch(self, **kwargs):
        return self.patch


def make_task(index):
    return SWERebenchTask(
        f"owner__repo-{index}", "owner/repo", "deadbeef", f"Fix issue {index}."
    )


class SWERebenchInferenceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.predictions = PredictionsWriter(
            self.root / "predictions.jsonl", model_name_or_path="model"
        )
        self.artifacts = RunArtifactsWriter(self.root)
        self.pipelines = []
        self.runtimes = []
        self.lock = threading.Lock()

    def tearDown(self):
        self.temp_dir.cleanup()

    def runner(self, fail_id=None, empty_id=None, max_workers=3, fail_fast=False):
        def pipeline_factory():
            pipeline = FakePipeline(fail_id=fail_id)
            with self.lock:
                self.pipelines.append(pipeline)
            return pipeline

        def runtime_factory(task, image):
            runtime = FakeRuntime(
                task,
                patch=(
                    "" if task.instance_id == empty_id else "diff --git a/a.py b/a.py\n"
                ),
            )
            with self.lock:
                self.runtimes.append(runtime)
            return runtime

        return SWERebenchInferenceRunner(
            pipeline_factory=pipeline_factory,
            runtime_factory=runtime_factory,
            image_resolver=InstanceImageResolver(allow_convention=True),
            predictions_writer=self.predictions,
            artifacts_writer=self.artifacts,
            max_workers=max_workers,
            fail_fast=fail_fast,
        )

    def test_parallel_tasks_use_fresh_pipeline_and_runtime(self):
        tasks = [make_task(index) for index in range(5)]
        summary = self.runner().run(tasks)
        self.assertEqual((summary.completed, summary.failed), (5, 0))
        self.assertEqual(len({id(item) for item in self.pipelines}), 5)
        self.assertEqual(len({id(item) for item in self.runtimes}), 5)
        self.assertTrue(all(item.closed for item in self.pipelines))
        self.assertTrue(all(item.is_closed for item in self.runtimes))
        self.assertCountEqual(
            [item.runtime_seen for item in self.pipelines],
            [task.instance_id for task in tasks],
        )
        records = [
            json.loads(line)
            for line in (self.root / "predictions.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(records), 5)

    def test_resume_skips_completed_tasks(self):
        self.predictions.write("owner__repo-0", "existing")
        summary = self.runner().run([make_task(0), make_task(1)])
        self.assertEqual(summary.skipped, 1)
        self.assertEqual(summary.completed, 1)

    def test_failures_and_empty_patches_are_recorded(self):
        summary = self.runner(fail_id="owner__repo-0", empty_id="owner__repo-1").run(
            [make_task(0), make_task(1), make_task(2)]
        )
        self.assertEqual(summary.completed, 1)
        self.assertEqual(summary.failed, 2)
        errors = [
            json.loads(line)
            for line in (self.root / "errors.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(errors), 2)
        records = [
            json.loads(line)
            for line in (self.root / "predictions.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(records), 3)
        patches = {record["instance_id"]: record["model_patch"] for record in records}
        self.assertEqual(patches["owner__repo-0"], "")
        self.assertEqual(patches["owner__repo-1"], "")
        self.assertTrue(patches["owner__repo-2"].startswith("diff --git"))

    def test_fail_fast_still_materializes_all_selected_predictions(self):
        tasks = [make_task(0), make_task(1)]
        with self.assertRaisesRegex(RuntimeError, "pipeline failed"):
            self.runner(fail_id="owner__repo-0", max_workers=1, fail_fast=True).run(
                tasks
            )

        records = [
            json.loads(line)
            for line in (self.root / "predictions.jsonl").read_text().splitlines()
        ]
        self.assertCountEqual(
            [record["instance_id"] for record in records],
            [task.instance_id for task in tasks],
        )
        patches = {record["instance_id"]: record["model_patch"] for record in records}
        self.assertEqual(patches["owner__repo-0"], "")


if __name__ == "__main__":
    unittest.main()
