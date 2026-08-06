"""Inference orchestration with one pipeline and runtime per task."""

import concurrent.futures as futures
import multiprocessing
import queue
import signal
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from src.agent_constructor.pipeline import Pipeline

from .data_types import SWERebenchTask
from .images import InstanceImage, InstanceImageResolver
from .predictions import PredictionsWriter, RunArtifactsWriter
from .prompt import SWERebenchPromptBuilder
from .runtime import RepositoryRuntime, bind_repository_runtime


@dataclass(frozen=True)
class InferenceSummary:
    submitted: int
    completed: int
    failed: int
    skipped: int


class SWERebenchInferenceRunner:
    """Generate patches without sharing mutable pipelines or task runtimes."""

    def __init__(
        self,
        *,
        pipeline_factory: Callable[[], Pipeline],
        runtime_factory: Callable[[SWERebenchTask, InstanceImage], RepositoryRuntime],
        image_resolver: InstanceImageResolver,
        predictions_writer: PredictionsWriter,
        artifacts_writer: RunArtifactsWriter,
        prompt_builder: SWERebenchPromptBuilder | None = None,
        max_workers: int = 1,
        fail_fast: bool = False,
        max_patch_chars: int = 1_000_000,
        task_timeout: int | None = None,
    ) -> None:
        if max_workers < 1 or max_patch_chars < 1:
            raise ValueError("Runner limits must be positive")
        self.pipeline_factory = pipeline_factory
        self.runtime_factory = runtime_factory
        self.image_resolver = image_resolver
        self.predictions_writer = predictions_writer
        self.artifacts_writer = artifacts_writer
        self.prompt_builder = prompt_builder or SWERebenchPromptBuilder()
        self.max_workers = max_workers
        self.fail_fast = fail_fast
        self.max_patch_chars = max_patch_chars
        if task_timeout is not None and task_timeout < 1:
            raise ValueError("task_timeout must be positive")
        self.task_timeout = task_timeout

    def run(self, tasks: Iterable[SWERebenchTask]) -> InferenceSummary:
        task_list = list(tasks)
        completed_ids = self.predictions_writer.completed_ids
        pending = [task for task in task_list if task.instance_id not in completed_ids]
        skipped = len(task_list) - len(pending)
        completed = 0
        failed = 0
        first_error: Exception | None = None

        if self.task_timeout is not None:
            return self._run_in_processes(task_list, pending, skipped)

        with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            submitted = {
                executor.submit(self._run_task, task): task for task in pending
            }
            for future in futures.as_completed(submitted):
                task = submitted[future]
                try:
                    future.result()
                    completed += 1
                except Exception as error:
                    failed += 1
                    self.artifacts_writer.write_error(
                        task.instance_id, "inference", error
                    )
                    # Preserve the benchmark denominator: every selected task gets a
                    # prediction, including infrastructure errors and empty patches.
                    self.predictions_writer.write(task.instance_id, "")
                    if self.fail_fast and first_error is None:
                        first_error = error
                        for other in submitted:
                            other.cancel()
        if first_error is not None:
            raise first_error
        return InferenceSummary(len(pending), completed, failed, skipped)

    def _run_in_processes(
        self,
        task_list: list[SWERebenchTask],
        pending: list[SWERebenchTask],
        skipped: int,
    ) -> InferenceSummary:
        """Run killable task processes; Linux fork preserves configured factories."""

        context = multiprocessing.get_context("fork")
        waiting = iter(pending)
        active: dict[Any, tuple[SWERebenchTask, Any, float]] = {}
        completed = failed = 0
        stop_submitting = False

        while active or not stop_submitting:
            while not stop_submitting and len(active) < self.max_workers:
                try:
                    task = next(waiting)
                except StopIteration:
                    stop_submitting = True
                    break
                result_queue = context.Queue(maxsize=1)
                process = context.Process(
                    target=self._process_entry,
                    args=(task, result_queue),
                    name=f"swe-rebench-{task.instance_id}",
                )
                process.start()
                active[process] = (task, result_queue, time.monotonic())

            for process, (task, result_queue, started) in list(active.items()):
                timed_out = time.monotonic() - started >= self.task_timeout
                queued_outcome = None
                if not timed_out:
                    try:
                        queued_outcome = result_queue.get_nowait()
                    except queue.Empty:
                        pass
                if process.is_alive() and not timed_out and queued_outcome is None:
                    continue
                if timed_out and process.is_alive():
                    process.terminate()
                    process.join(timeout=30)
                    if process.is_alive():
                        process.kill()
                        process.join()
                    error = TimeoutError(
                        f"Inference exceeded task timeout of {self.task_timeout} seconds"
                    )
                    outcome = ("error", type(error).__name__, str(error))
                else:
                    process.join()
                    if queued_outcome is not None:
                        outcome = queued_outcome
                    else:
                        try:
                            outcome = result_queue.get_nowait()
                        except queue.Empty:
                            outcome = (
                                "error",
                                "WorkerProcessError",
                                f"Worker exited with code {process.exitcode} without a result",
                            )
                result_queue.close()
                del active[process]
                if outcome[0] == "ok":
                    self.predictions_writer.write(task.instance_id, outcome[1])
                    self.artifacts_writer.write_patch_artifact(
                        task.instance_id, outcome[2]
                    )
                    completed += 1
                else:
                    error = RuntimeError(f"{outcome[1]}: {outcome[2]}")
                    self.artifacts_writer.write_error(
                        task.instance_id, "inference", error
                    )
                    self.predictions_writer.write(task.instance_id, "")
                    failed += 1
                    if self.fail_fast:
                        stop_submitting = True
                        for other in active:
                            other.terminate()
                break
            else:
                if active:
                    time.sleep(0.05)
                elif stop_submitting:
                    break

        # Preserve the denominator for tasks not submitted after fail-fast.
        completed_ids = self.predictions_writer.completed_ids
        for task in task_list:
            if task.instance_id not in completed_ids:
                self.predictions_writer.write(task.instance_id, "")
                failed += 1
        return InferenceSummary(len(pending), completed, failed, skipped)

    def _process_entry(self, task: SWERebenchTask, result_queue: Any) -> None:
        runtime: RepositoryRuntime | None = None

        def terminate(_signum, _frame):
            if runtime is not None:
                runtime.close()
            raise TimeoutError("Inference worker terminated by parent")

        signal.signal(signal.SIGTERM, terminate)
        pipeline = None
        try:
            image = self.image_resolver.resolve(task)
            pipeline = self.pipeline_factory()
            runtime = self.runtime_factory(task, image)
            with runtime:
                with bind_repository_runtime(runtime):
                    prompt = self.prompt_builder.build(task, runtime.workdir)
                    pipeline.run(prompt)
                    patch = runtime.get_patch(max_output_chars=self.max_patch_chars)
                    summary = runtime.get_diff(stat_only=True, max_output_chars=30_000)
            if not patch.strip():
                raise RuntimeError("Agent produced an empty model patch")
            changed_files = sorted(
                {
                    line.split(" ", 3)[2][2:]
                    for line in patch.splitlines()
                    if line.startswith("diff --git a/") and len(line.split(" ", 3)) >= 3
                }
            )
            result_queue.put(
                (
                    "ok",
                    patch,
                    {
                        "patch_size": len(patch.encode("utf-8")),
                        "changed_files": changed_files,
                        "diff_summary": summary,
                    },
                )
            )
        except BaseException as error:
            result_queue.put(("error", type(error).__name__, str(error)))
        finally:
            if pipeline is not None:
                pipeline.close()

    def _run_task(self, task: SWERebenchTask) -> None:
        image = self.image_resolver.resolve(task)
        pipeline = self.pipeline_factory()
        try:
            runtime = self.runtime_factory(task, image)
            with runtime:
                with bind_repository_runtime(runtime):
                    prompt = self.prompt_builder.build(task, runtime.workdir)
                    pipeline.run(prompt)
                    patch = runtime.get_patch(max_output_chars=self.max_patch_chars)
            if not patch.strip():
                raise RuntimeError("Agent produced an empty model patch")
            self.predictions_writer.write(task.instance_id, patch)
            self.artifacts_writer.write_patch_artifact(
                task.instance_id,
                {
                    "patch_size": len(patch.encode("utf-8")),
                    "changed_files": sorted(
                        {
                            line.split(" ", 3)[2][2:]
                            for line in patch.splitlines()
                            if line.startswith("diff --git a/")
                            and len(line.split(" ", 3)) >= 3
                        }
                    ),
                },
            )
        finally:
            pipeline.close()
