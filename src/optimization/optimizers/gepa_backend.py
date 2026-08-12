"""Bridge to the upstream ``gepa`` package (github.com/gepa-ai/gepa).

GEPA (Agrawal et al., 2025 — *Reflective Prompt Evolution Can Outperform
Reinforcement Learning*) evolves prompts by reflecting on execution traces and
keeping a Pareto frontier over training instances rather than a single best
candidate. We adapt rather than reimplement it: this module is the only place
that touches the upstream API, so a signature change is a local fix.

⚠ The adapter is written against GEPA's documented protocol —
``evaluate(batch, candidate, capture_traces)`` returning an object with
``outputs`` / ``scores`` / ``trajectories``, plus ``make_reflective_dataset``.
It has not been executed against an installed copy here (the package is not in
pyproject.toml and the server has no outbound network). Run
``python optimize_prompts.py --check`` first: it reports what the installed
version actually exposes instead of failing deep inside a rollout.
"""

import logging

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.optimization.candidates import Candidate
from src.optimization.optimizers.base import OptimizationResult

logger = logging.getLogger(__name__)


INSTALL_HINT = (
    "the `gepa` package is not installed.\n"
    "  pip install gepa\n"
    "If the server has no outbound network (the fastembed failure suggests it "
    "may not), run with `--optimizer random` — same rollout, same artifacts, "
    "no reflective search."
)


def require_gepa():
    try:
        import gepa                                          # noqa: F401
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return gepa


@dataclass
class _FallbackEvaluationBatch:
    """Duck-typed stand-in used when the upstream dataclass moves."""
    outputs: List[Any] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    trajectories: Optional[List[Any]] = None


def _evaluation_batch_cls():
    """Locate upstream's EvaluationBatch, else fall back to the local shape."""
    for module_path in ("gepa.core.adapter", "gepa.adapter", "gepa"):
        try:
            module = __import__(module_path, fromlist=["EvaluationBatch"])
        except ImportError:
            continue
        cls = getattr(module, "EvaluationBatch", None)
        if cls is not None:
            return cls
    logger.warning("gepa.EvaluationBatch not found; using a duck-typed stand-in")
    return _FallbackEvaluationBatch


def check_installation() -> Dict[str, Any]:
    """What the installed gepa actually exposes — run before a long job."""
    report: Dict[str, Any] = {"installed": False}
    try:
        gepa = require_gepa()
    except ImportError as exc:
        report["error"] = str(exc)
        return report
    report["installed"] = True
    report["version"] = getattr(gepa, "__version__", "unknown")
    report["has_optimize"] = hasattr(gepa, "optimize")
    report["evaluation_batch"] = _evaluation_batch_cls().__name__
    report["top_level"] = sorted(n for n in dir(gepa) if not n.startswith("_"))
    return report


class _RagGuideAdapter:
    """Turns our rollout into what GEPA expects from a system under test."""

    def __init__(self, score_fn, log):
        self.score_fn = score_fn
        self.log = log
        self.batch_cls = _evaluation_batch_cls()
        self.metric_calls = 0
        self.last_result = None

    def evaluate(self, batch: Sequence, candidate: Candidate,
                 capture_traces: bool = False):
        task_ids = list(batch)
        try:
            result = self.score_fn(candidate, task_ids)
        except ValueError as exc:
            # A malformed candidate (dropped $placeholder) scores zero rather
            # than killing the search.
            self.log(f"[gepa] invalid candidate: {exc}")
            return self.batch_cls(
                outputs=[""] * len(task_ids),
                scores=[0.0] * len(task_ids),
                trajectories=None,
            )

        self.metric_calls += 1
        self.last_result = result
        scores = result.scores
        feedback = result.feedback()
        self.log(f"[gepa] rollout {self.metric_calls}: score={result.mean:.3f}")

        outputs = [feedback.get(pid, "") for pid in task_ids]
        trajectories = None
        if capture_traces:
            trajectories = [
                {"task_id": pid, "score": scores.get(pid, 0.0),
                 "feedback": feedback.get(pid, "")}
                for pid in task_ids
            ]
        return self.batch_cls(
            outputs=outputs,
            scores=[scores.get(pid, 0.0) for pid in task_ids],
            trajectories=trajectories,
        )

    def make_reflective_dataset(self, candidate: Candidate, eval_batch,
                                components_to_update: Sequence[str]
                                ) -> Dict[str, List[Dict]]:
        """Failures first — the reflection LM should read what went wrong."""
        trajectories = getattr(eval_batch, "trajectories", None) or []
        records = []
        for entry in trajectories:
            records.append({
                "Inputs": f"DS1000 task {entry.get('task_id')}",
                "Generated Outputs": entry.get("feedback", ""),
                "Feedback": entry.get("feedback", ""),
            })
        records.sort(key=lambda r: "FAILED" not in r["Feedback"])
        return {component: records for component in components_to_update}


class GepaOptimizer:
    """PromptOptimizer backed by the upstream package."""

    name = "gepa"

    def __init__(self, reflection_lm: Callable[[str], str], seed: int = 7,
                 valset_ids: Optional[Sequence] = None, **optimize_kwargs):
        self.reflection_lm = reflection_lm
        self.seed = seed
        self.valset_ids = valset_ids
        self.optimize_kwargs = optimize_kwargs

    def optimize(self, seed: Candidate, score, train_ids: Sequence, budget: int,
                 log: Optional[Callable[[str], None]] = None) -> OptimizationResult:
        gepa = require_gepa()
        log = log or (lambda message: None)

        adapter = _RagGuideAdapter(score_fn=score, log=log)
        baseline = score(seed, train_ids)
        log(f"[gepa] seed score={baseline.mean:.3f}")

        raw = gepa.optimize(
            seed_candidate=dict(seed),
            trainset=list(train_ids),
            valset=list(self.valset_ids or train_ids),
            adapter=adapter,
            reflection_lm=self.reflection_lm,
            max_metric_calls=budget,
            **self.optimize_kwargs,
        )

        best = _extract_best(raw) or dict(seed)
        best_result = score(best, train_ids)
        log(f"[gepa] best score={best_result.mean:.3f} "
            f"(seed {baseline.mean:.3f})")

        return OptimizationResult(
            best=best,
            best_score=best_result.mean,
            seed_score=baseline.mean,
            history=[{"metric_calls": adapter.metric_calls}],
            backend=self.name,
            metric_calls=adapter.metric_calls,
        )


def _extract_best(raw) -> Optional[Candidate]:
    """Upstream has renamed this attribute before; try the known spellings."""
    for attribute in ("best_candidate", "best_candidate_", "best", "candidate"):
        value = getattr(raw, attribute, None)
        if isinstance(value, dict) and value:
            return dict(value)
    if isinstance(raw, dict) and raw:
        return dict(raw)
    logger.warning("could not read the best candidate off %r", type(raw))
    return None
