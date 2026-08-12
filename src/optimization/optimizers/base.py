"""The optimizer contract.

Everything above this line is optimizer-agnostic: a candidate is a dict of
prompt texts, and scoring one is a call to `evaluate`. A backend only has to
turn a seed candidate plus a budget into a better candidate.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from src.optimization.candidates import Candidate


@dataclass
class OptimizationResult:
    best: Candidate
    best_score: float
    seed_score: float
    history: List[Dict] = field(default_factory=list)
    backend: str = ""
    metric_calls: int = 0

    @property
    def improvement(self) -> float:
        return self.best_score - self.seed_score


class ScoreFn(Protocol):
    """Scores a candidate on the given tasks.

    Returns per-task scores and per-task natural-language feedback — the
    feedback is what separates reflective optimization from random search.
    """

    def __call__(self, candidate: Candidate, task_ids: Sequence) -> "RolloutView":
        ...


class RolloutView(Protocol):
    @property
    def mean(self) -> float: ...
    @property
    def scores(self) -> Dict[object, float]: ...
    def feedback(self) -> Dict[object, str]: ...


class PromptOptimizer(Protocol):
    name: str

    def optimize(
        self,
        seed: Candidate,
        score: ScoreFn,
        train_ids: Sequence,
        budget: int,
        log: Optional[Callable[[str], None]] = None,
    ) -> OptimizationResult:
        ...
