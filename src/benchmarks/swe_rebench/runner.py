"""Inference orchestration with one pipeline and runtime per task."""

import logging
import multiprocessing
import queue
import signal
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
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


class EmptyModelPatchError(RuntimeError):
    """Raised when an agent finishes without changing the task checkout."""


class SWERebenchInferenceRunner:
    """Generate patches without sharing mutable pipelines or task runtimes."""

    def __init__(
        self,
        *,
        pipeline_factory: Callable[[], Pipeline],
        runtime_factory: Callable[[SWERebenchTask, InstanceImage], RepositoryRuntime],
        image_resolver: InstanceImageResolver,
        predictions_writer: PredictionsWriter,
        resolved_images: Mapping[str, InstanceImage] | None = None,
        artifacts_writer: RunArtifactsWriter,
        prompt_builder: SWERebenchPromptBuilder | None = None,
        max_workers: int = 1,
        fail_fast: bool = False,
        max_patch_chars: int = 1_000_000,
        task_timeout: int = 3_600,
    ) -> None:
        if max_workers < 1 or max_patch_chars < 1:
            raise ValueError("Runner limits must be positive")
        self.pipeline_factory = pipeline_factory
        self.runtime_factory = runtime_factory
        self.image_resolver = image_resolver
        self.resolved_images = dict(resolved_images or {})
        if not all(
            isinstance(instance_id, str)
            and instance_id.strip()
            and isinstance(image, InstanceImage)
            for instance_id, image in self.resolved_images.items()
        ):
            raise TypeError("resolved_images must map instance IDs to InstanceImage")
        self.predictions_writer = predictions_writer
        self.artifacts_writer = artifacts_writer
        self.prompt_builder = prompt_builder or SWERebenchPromptBuilder()
        self.max_workers = max_workers
        self.fail_fast = fail_fast
        self.max_patch_chars = max_patch_chars
        if task_timeout < 1:
            raise ValueError("task_timeout must be positive")
        self.task_timeout = task_timeout

    def run(self, tasks: Iterable[SWERebenchTask]) -> InferenceSummary:
        task_list = list(tasks)
        completed_ids = self.predictions_writer.completed_ids
        pending = [task for task in task_list if task.instance_id not in completed_ids]
        skipped = len(task_list) - len(pending)
        return self._run_in_processes(task_list, pending, skipped)

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
                    args=(task, self.resolved_images.get(task.instance_id), result_queue),
                    name=f"swe-rebench-{task.instance_id}",
                )
                process.start()
                logging.info(
                    "STAGE\tTASK_SUBMITTED\tinstance_id=%s\tpid=%s",
                    task.instance_id,
                    process.pid,
                )
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
                    logging.error(
                        "STAGE\tTASK_TIMEOUT\tinstance_id=%s\ttimeout_seconds=%s",
                        task.instance_id,
                        self.task_timeout,
                    )
                    process.terminate()
                    process.join(timeout=30)
                    if process.is_alive():
                        process.kill()
                        process.join()
                    error = TimeoutError(
                        f"Inference exceeded task timeout of {self.task_timeout} seconds"
                    )
                    outcome = (
                        "error",
                        type(error).__name__,
                        str(error),
                        None,
                        {"termination_reason": "task_timeout"},
                        time.monotonic() - started,
                    )
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
                                None,
                                {
                                    "termination_reason": "worker_exited_without_result",
                                    "worker_exit_code": process.exitcode,
                                },
                                time.monotonic() - started,
                            )
                result_queue.close()
                del active[process]
                if outcome[0] == "ok":
                    self.predictions_writer.write(task.instance_id, outcome[1])
                    self.artifacts_writer.write_patch_artifact(
                        task.instance_id, outcome[2]
                    )
                    completed += 1
                    logging.info(
                        "STAGE\tTASK_COMPLETED\tinstance_id=%s\tduration_seconds=%.3f",
                        task.instance_id,
                        time.monotonic() - started,
                    )
                else:
                    error = RuntimeError(outcome[2])
                    self.artifacts_writer.write_error(
                        task.instance_id,
                        "inference",
                        error,
                        duration_seconds=outcome[5],
                        error_type=outcome[1],
                        traceback_text=outcome[3],
                        details=outcome[4],
                    )
                    self.predictions_writer.write(task.instance_id, "")
                    failed += 1
                    logging.error(
                        "STAGE\tTASK_FAILED\tinstance_id=%s\terror_type=%s\tmessage=%s",
                        task.instance_id,
                        outcome[1],
                        outcome[2],
                    )
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

    def _process_entry(
        self,
        task: SWERebenchTask,
        image: InstanceImage | None,
        result_queue: Any,
    ) -> None:
        runtime: RepositoryRuntime | None = None
        started = time.monotonic()
        agent_result: Any = None
        diagnostics: dict[str, Any] = {}
        logging.info(
            "STAGE\tWORKER_STARTED\tinstance_id=%s\tpid=%s",
            task.instance_id,
            multiprocessing.current_process().pid,
        )

        def terminate(_signum, _frame):
            if runtime is not None:
                runtime.close()
            raise TimeoutError("Inference worker terminated by parent")

        signal.signal(signal.SIGTERM, terminate)
        pipeline = None
        try:
            if image is None:
                image = self.image_resolver.resolve(task)
            logging.info(
                "STAGE\tWORKER_IMAGE_READY\tinstance_id=%s\timage=%s",
                task.instance_id,
                image.name,
            )
            pipeline = self.pipeline_factory()
            logging.info("STAGE\tPIPELINE_CREATED\tinstance_id=%s", task.instance_id)
            runtime = self.runtime_factory(task, image)
            container = getattr(runtime, "container", None)
            logging.info(
                "STAGE\tCONTAINER_STARTED\tinstance_id=%s\tcontainer_id=%s\tcontainer_name=%s",
                task.instance_id,
                getattr(container, "id", "unknown"),
                getattr(container, "name", "unknown"),
            )
            with runtime:
                with bind_repository_runtime(runtime):
                    prompt = self.prompt_builder.build(task, runtime.workdir)
                    logging.info(
                        "STAGE\tPROMPT_BUILT\tinstance_id=%s\tchars=%s",
                        task.instance_id,
                        len(prompt),
                    )
                    pipeline_started = time.monotonic()
                    logging.info(
                        "STAGE\tPIPELINE_STARTED\tinstance_id=%s", task.instance_id
                    )
                    agent_result = pipeline.run(prompt)
                    agent_termination_reason = self._agent_termination_reason(
                        agent_result
                    )
                    if agent_termination_reason is not None:
                        logging.warning(
                            "STAGE\tAGENT_TERMINATED\tinstance_id=%s\treason=%s\t"
                            "patch_will_be_collected=true",
                            task.instance_id,
                            agent_termination_reason,
                        )
                    logging.info(
                        "STAGE\tPIPELINE_FINISHED\tinstance_id=%s\tduration_seconds=%.3f",
                        task.instance_id,
                        time.monotonic() - pipeline_started,
                    )
                    patch = runtime.get_patch(max_output_chars=self.max_patch_chars)
                    logging.info(
                        "STAGE\tPATCH_COLLECTED\tinstance_id=%s\tchars=%s",
                        task.instance_id,
                        len(patch),
                    )
                    summary = runtime.get_diff(stat_only=True, max_output_chars=30_000)
                    diagnostics = self._diagnostics(
                        task=task,
                        image=image,
                        runtime=runtime,
                        agent_result=agent_result,
                        diff_summary=summary,
                    )
                    if agent_termination_reason is not None:
                        diagnostics["agent_termination_reason"] = (
                            agent_termination_reason
                        )
                    if not patch.strip():
                        raise EmptyModelPatchError(
                            "Agent finished without modifying the repository. "
                            "See details.agent_result, details.git_status, and "
                            "details.diff_summary for the termination context."
                        )
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
                        **(
                            {"agent_termination_reason": agent_termination_reason}
                            if agent_termination_reason is not None
                            else {}
                        ),
                    },
                )
            )
        except BaseException as error:
            logging.exception(
                "STAGE\tWORKER_FAILED\tinstance_id=%s\terror_type=%s",
                task.instance_id,
                type(error).__name__,
            )
            if image is not None:
                diagnostics.setdefault("image", image.name)
            diagnostics.setdefault("agent_result", self._bounded_text(agent_result))
            result_queue.put(
                (
                    "error",
                    type(error).__name__,
                    str(error),
                    traceback.format_exc(),
                    diagnostics,
                    time.monotonic() - started,
                )
            )
        finally:
            if pipeline is not None:
                pipeline.close()
                logging.info("STAGE\tPIPELINE_CLOSED\tinstance_id=%s", task.instance_id)
            logging.info(
                "STAGE\tWORKER_FINISHED\tinstance_id=%s\tduration_seconds=%.3f",
                task.instance_id,
                time.monotonic() - started,
            )
            logging.shutdown()

    @classmethod
    def _diagnostics(
        cls,
        *,
        task: SWERebenchTask,
        image: InstanceImage,
        runtime: RepositoryRuntime,
        agent_result: Any,
        diff_summary: str,
    ) -> dict[str, Any]:
        status = runtime.run_command(
            "git status --short", timeout=30, max_output_chars=10_000
        )
        return {
            "repo": task.repo,
            "base_commit": task.base_commit,
            "image": image.name,
            "workdir": runtime.workdir,
            "agent_result": cls._bounded_text(agent_result),
            "git_status": status.stdout.strip() or "[clean]",
            "git_status_stderr": status.stderr.strip(),
            "git_status_exit_code": status.exit_code,
            "diff_summary": diff_summary,
        }

    @staticmethod
    def _agent_termination_reason(agent_result: Any) -> str | None:
        if not isinstance(agent_result, str):
            return None
        normalized = agent_result.strip().lower()
        if normalized.startswith("error: maximum iterations reached"):
            return "max_iterations"
        if normalized.startswith(
            "error: agent failed to produce a valid response"
        ):
            return "invalid_response_retry_limit"
        if normalized.startswith("error: could not generate final answer"):
            return "final_answer_generation_failed"
        return None

    @staticmethod
    def _bounded_text(value: Any, limit: int = 4_000) -> str | None:
        if value is None:
            return None
        text = str(value)
        if len(text) <= limit:
            return text
        return text[:limit] + "\n[truncated]"
