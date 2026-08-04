"""Inference orchestration with one pipeline and runtime per task."""

import concurrent.futures as futures
from collections.abc import Callable, Iterable
from dataclasses import dataclass

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

    def run(self, tasks: Iterable[SWERebenchTask]) -> InferenceSummary:
        task_list = list(tasks)
        completed_ids = self.predictions_writer.completed_ids
        pending = [task for task in task_list if task.instance_id not in completed_ids]
        skipped = len(task_list) - len(pending)
        completed = 0
        failed = 0

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
                    if self.fail_fast:
                        for other in submitted:
                            other.cancel()
                        raise
        return InferenceSummary(len(pending), completed, failed, skipped)

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
        finally:
            pipeline.close()
