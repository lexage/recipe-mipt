"""Random-mutation baseline — the control every optimizer must beat.

Same interface, same budget, same rollout: it asks the reflection LM for a
rewrite *without* showing it any execution feedback, and keeps a change only
when the score improves. If GEPA cannot beat this, the reflective feedback is
not doing any work and the result is not worth reporting.
"""

import random

from typing import Callable, Dict, List, Optional, Sequence

from src.optimization.candidates import Candidate
from src.optimization.optimizers.base import OptimizationResult


REWRITE_INSTRUCTION = (
    "You are refining an instruction given to a documentation-writing model.\n"
    "Rewrite the instruction below to be clearer and more specific. Keep every "
    "$placeholder exactly as it appears — dropping one breaks the system. "
    "Reply with the rewritten instruction only, no commentary.\n\n"
    "--- CURRENT INSTRUCTION ---\n{text}\n--- END ---"
)


class RandomMutationOptimizer:
    name = "random"

    def __init__(self, reflection_lm: Callable[[str], str], seed: int = 7):
        self.reflection_lm = reflection_lm
        self.rng = random.Random(seed)

    def optimize(
        self,
        seed: Candidate,
        score,
        train_ids: Sequence,
        budget: int,
        log: Optional[Callable[[str], None]] = None,
    ) -> OptimizationResult:
        log = log or (lambda message: None)

        base = score(seed, train_ids)
        best, best_score = dict(seed), base.mean
        history: List[Dict] = []
        calls = 1
        log(f"[random] seed score={best_score:.3f}")

        keys = sorted(seed)
        while calls < budget and keys:
            key = self.rng.choice(keys)
            proposal = self.reflection_lm(REWRITE_INSTRUCTION.format(text=best[key]))
            if not proposal or not proposal.strip():
                continue

            trial = dict(best)
            trial[key] = proposal.strip() + ("\n" if not proposal.endswith("\n") else "")
            try:
                result = score(trial, train_ids)
            except ValueError as exc:            # e.g. a dropped $placeholder
                log(f"[random] rejected mutation of {key}: {exc}")
                history.append({"key": key, "score": None, "error": str(exc)})
                calls += 1
                continue

            calls += 1
            kept = result.mean > best_score
            history.append({"key": key, "score": result.mean, "kept": kept})
            log(f"[random] {key}: {result.mean:.3f} "
                f"({'kept' if kept else 'discarded'})")
            if kept:
                best, best_score = trial, result.mean

        return OptimizationResult(
            best=best,
            best_score=best_score,
            seed_score=base.mean,
            history=history,
            backend=self.name,
            metric_calls=calls,
        )
