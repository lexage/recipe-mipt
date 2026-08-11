import json
import logging
import tempfile
import threading
import time
import unittest
from pathlib import Path

from run_swe_rebench import resolve_run_logs_path
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
from src.utils.loggers import create_logging


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


class SlowPipeline(FakePipeline):
    def run(self, prompt):
        time.sleep(5)


class LoggingPipeline(FakePipeline):
    def run(self, prompt):
        logging.info("AGENT_LOG_MARKER")
        return super().run(prompt)


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
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
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

    def test_process_task_timeout_materializes_empty_prediction(self):
        def runtime_factory(task, image):
            return FakeRuntime(task)

        runner = SWERebenchInferenceRunner(
            pipeline_factory=SlowPipeline,
            runtime_factory=runtime_factory,
            image_resolver=InstanceImageResolver(allow_convention=True),
            predictions_writer=self.predictions,
            artifacts_writer=self.artifacts,
            max_workers=1,
            task_timeout=1,
        )

        summary = runner.run([make_task(0)])

        self.assertEqual(summary.failed, 1)
        record = json.loads((self.root / "predictions.jsonl").read_text().strip())
        self.assertEqual(record["model_patch"], "")

    def test_process_worker_returns_patch_and_artifacts(self):
        def runtime_factory(task, image):
            return FakeRuntime(task)

        runner = SWERebenchInferenceRunner(
            pipeline_factory=FakePipeline,
            runtime_factory=runtime_factory,
            image_resolver=InstanceImageResolver(allow_convention=True),
            predictions_writer=self.predictions,
            artifacts_writer=self.artifacts,
            max_workers=1,
            task_timeout=10,
        )

        summary = runner.run([make_task(0)])

        self.assertEqual(summary.completed, 1)
        self.assertTrue((self.root / "patches.jsonl").is_file())

    def test_empty_patch_error_contains_agent_and_repository_diagnostics(self):
        def runtime_factory(task, image):
            return FakeRuntime(task, patch="")

        runner = SWERebenchInferenceRunner(
            pipeline_factory=FakePipeline,
            runtime_factory=runtime_factory,
            image_resolver=InstanceImageResolver(allow_convention=True),
            predictions_writer=self.predictions,
            artifacts_writer=self.artifacts,
            max_workers=1,
            task_timeout=10,
        )

        summary = runner.run([make_task(0)])

        self.assertEqual(summary.failed, 1)
        error = json.loads((self.root / "errors.jsonl").read_text())
        self.assertEqual(error["error_type"], "EmptyModelPatchError")
        self.assertEqual(error["details"]["agent_result"], "ignored final answer")
        self.assertEqual(error["details"]["git_status"], "[clean]")
        self.assertIn("Traceback", error["traceback"])

    def test_swe_rebench_runs_write_to_separate_log_directories(self):
        logs = self.root / "logs"
        first_logs = resolve_run_logs_path(str(logs), "run-one")
        second_logs = resolve_run_logs_path(str(logs), "run-two")

        create_logging(str(first_logs), n_workers=1, route=True)
        logging.info("FIRST_RUN_MARKER")
        create_logging(str(second_logs), n_workers=1, route=True)
        logging.info("SECOND_RUN_MARKER")
        for handler in logging.getLogger().handlers:
            handler.flush()

        first_content = (first_logs / "log_main.log").read_text()
        second_content = (second_logs / "log_main.log").read_text()
        self.assertIn("FIRST_RUN_MARKER", first_content)
        self.assertNotIn("SECOND_RUN_MARKER", first_content)
        self.assertIn("SECOND_RUN_MARKER", second_content)
        self.assertNotIn("FIRST_RUN_MARKER", second_content)

    def test_process_worker_and_agent_messages_are_written_to_log_file(self):
        def runtime_factory(task, image):
            return FakeRuntime(task)

        logs = self.root / "logs"
        create_logging(str(logs), tag="test", n_workers=1, route=True)
        runner = SWERebenchInferenceRunner(
            pipeline_factory=LoggingPipeline,
            runtime_factory=runtime_factory,
            image_resolver=InstanceImageResolver(allow_convention=True),
            predictions_writer=self.predictions,
            artifacts_writer=self.artifacts,
            max_workers=1,
            task_timeout=10,
        )

        runner.run([make_task(0)])
        for handler in logging.getLogger().handlers:
            handler.flush()
        process_logs = list(logs.glob("log_test_process_*.log"))
        self.assertEqual(len(process_logs), 1)
        content = process_logs[0].read_text()

        self.assertIn("AGENT_LOG_MARKER", content)
        self.assertIn("STAGE\tPIPELINE_STARTED", content)
        self.assertIn("STAGE\tPATCH_COLLECTED", content)


if __name__ == "__main__":
    unittest.main()
