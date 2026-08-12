"""Scoring a candidate prompt set — one rollout.

The rollout runs **the pipeline itself**: `pipeline.run(task.prompt)`, then the
real DS1000 test. Nothing here knows whether the pipeline has one solver or a
panel of them with an aggregator, whether it retrieves, or which prompts belong
to which component — it scores whatever `PipelineBuilder` assembled. That is
the only way the number means anything for pipelines other than SIMPLE.

Two ways a candidate reaches the components:

* **rebuild** (default) — prompt overrides are installed process-wide and the
  pipeline is rebuilt, so prompts consumed at *construction* time take effect.
  `RagGuideGenerator` is in this class: its `generate()` runs inside
  `SimplePipeline.__init__`, so mutating it afterwards would change nothing.
* **reuse** (`reuse_pipeline=True`) — mutate the `Prompt` objects of an
  already-built pipeline in place and re-run. Only correct when every optimized
  prompt is consumed at query time (a solver's system prompt, a query-time
  filter). Much faster: no regeneration, no re-indexing.

Rebuild cost is controlled by the caller shrinking the corpus
(`data_base.max_docs`) and capping generator calls (`generator.limit`). Each
rebuild MUST get a fresh vector-db directory: Qdrant skips chunk ids it already
holds, so reusing a directory would silently index none of the new documents
and every candidate would score identically.
"""

import logging
import shutil

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from src.agent_constructor.prompts import clear_overrides, set_overrides
from src.optimization import candidates as cand
from src.utils.retrieval_log import take as take_retrieval_record
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)

MAX_FEEDBACK_CHARS = 500


def ds1000_scorer(timeout: float = 120.0):
    """Default scorer: run the task's real DS1000 test on the pipeline's answer.

    Returns ``(score, result_text, code)``. The result text is what makes the
    feedback useful — an execution error rather than a bare 0.
    """

    def score(task, answer: str):
        # Lazy: pulls pandas/tqdm, which the light modules must not require.
        from src.benchmarks.ds1000 import DS1000
        from src.benchmarks.ds1000 import execution

        code = DS1000._postprocess(answer or "")
        program = (
            task.code_context + "\n"
            + f"code = {repr(code)}\n"
            + "test_execution(code)\n"
            + ("test_string(code)\n" if "test_string(" in task.code_context else "\n")
        )
        verdict, _meta = execution.check_correctness(
            program, timeout=timeout, metadata=task.metadata
        )
        return (int(verdict.get("score", 0)),
                str(verdict.get("result", "")),
                code)

    return score


@dataclass
class TaskOutcome:
    problem_id: object
    score: int
    result: str
    answer: str = ""
    retrieved: List[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        context = ", ".join(self.retrieved) or "nothing retrieved"
        if self.score == 1:
            return f"PASSED. Context: {context}"
        return (
            f"FAILED. Context: {context}\n"
            f"Execution result: {self.result[:MAX_FEEDBACK_CHARS]}\n"
            f"Model answer (start): {self.answer[:MAX_FEEDBACK_CHARS]}"
        )


@dataclass
class RolloutResult:
    outcomes: Dict[object, TaskOutcome]
    note: str = ""

    @property
    def scores(self) -> Dict[object, float]:
        return {pid: float(o.score) for pid, o in self.outcomes.items()}

    @property
    def mean(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.score for o in self.outcomes.values()) / len(self.outcomes)

    def feedback(self) -> Dict[object, str]:
        return {pid: o.as_feedback() for pid, o in self.outcomes.items()}


class PipelineEvaluator:
    """Runs a candidate through a real pipeline and scores it on DS1000.

    `build_pipeline(vdb_path)` must return a freshly built pipeline writing its
    vector index to `vdb_path`; the evaluator calls it once per rollout (unless
    `reuse_pipeline`) and closes it afterwards — Qdrant's local mode holds an
    exclusive lock, so two live pipelines cannot share a directory.
    """

    def __init__(
        self,
        build_pipeline: Callable[[str], object],
        dataset,
        scratch_dir: str,
        num_workers: int = 4,
        timeout: float = 120.0,
        score_answer: Optional[Callable] = None,
        reuse_pipeline: bool = False,
        keep_scratch: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ):
        self.build_pipeline = build_pipeline
        self.dataset = dataset
        self.scratch_dir = scratch_dir
        self.num_workers = max(1, int(num_workers))
        self.timeout = float(timeout)
        # Injectable so the rollout is not welded to DS1000 — and so tests can
        # score in-process instead of spawning the execution harness.
        self.score_answer = score_answer or ds1000_scorer(timeout)
        self.reuse_pipeline = bool(reuse_pipeline)
        self.keep_scratch = bool(keep_scratch)
        self.progress = progress or (lambda message: None)
        self.n_rollouts = 0
        self._pipeline = None          # only used when reuse_pipeline

    # ------------------------------------------------------------------ setup
    def _pipeline_for(self, candidate: cand.Candidate, tag: str):
        """A pipeline with `candidate` in effect, plus its cleanup callback."""
        if self.reuse_pipeline:
            if self._pipeline is None:
                set_overrides(candidate)
                self._pipeline = self.build_pipeline(
                    f"{self.scratch_dir}/vdb_reused"
                )
                clear_overrides()
            cand.apply(cand.from_pipeline(self._pipeline), candidate)
            return self._pipeline, (lambda: None)

        vdb_path = f"{self.scratch_dir}/vdb_{tag}"
        set_overrides(candidate)
        try:
            pipeline = self.build_pipeline(vdb_path)
        finally:
            clear_overrides()

        def cleanup():
            try:
                pipeline.close()
            except Exception as exc:                      # noqa: BLE001
                logger.warning("pipeline.close() failed: %s", exc)
            if not self.keep_scratch:
                shutil.rmtree(vdb_path, ignore_errors=True)

        return pipeline, cleanup

    # --------------------------------------------------------------- rollout
    def evaluate(self, candidate: cand.Candidate, task_ids: Sequence) -> RolloutResult:
        self.n_rollouts += 1
        tag = f"{self.n_rollouts:03d}"
        self.progress(f"rollout {tag}: building pipeline"
                      if not self.reuse_pipeline else f"rollout {tag}: reusing pipeline")
        pipeline, cleanup = self._pipeline_for(candidate, tag)

        tasks = [self.dataset[pid] for pid in task_ids]
        tracker = get_active()

        def solve(task):
            set_active(tracker)
            try:
                answer = pipeline.run(task.prompt)
            except Exception as exc:                      # one bad task != dead rollout
                logger.warning("pipeline.run failed for %s: %s",
                               task.metadata.get("problem_id"), exc)
                return "", []
            record = take_retrieval_record()
            titles = [
                (chunk.metadata or {}).get("doc_name") or str(chunk.doc_id)
                for chunk in record.get("chunks", [])
            ]
            return answer, titles

        try:
            self.progress(f"rollout {tag}: running {len(tasks)} tasks")
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                answers = list(pool.map(solve, tasks))
        finally:
            cleanup()

        outcomes: Dict[object, TaskOutcome] = {}
        for index, task in enumerate(tasks):
            answer, titles = answers[index]
            score, result, code = self.score_answer(task, answer)
            pid = task.metadata.get("problem_id")
            outcomes[pid] = TaskOutcome(
                problem_id=pid,
                score=int(score),
                result=str(result),
                answer=code,
                retrieved=titles[:5],
            )

        result = RolloutResult(outcomes=outcomes)
        self.progress(f"rollout {tag}: score={result.mean:.3f}")
        return result

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.close()
            except Exception as exc:                      # noqa: BLE001
                logger.warning("pipeline.close() failed: %s", exc)
            self._pipeline = None
