"""Optimizer backends, resolved by name.

Adding one: implement `PromptOptimizer` (see base.py) and register a factory
here. Nothing above this package changes — the candidate plumbing, the rollout
and the artifacts are shared.

Imports are lazy so a missing optional dependency only breaks the backend that
needs it: `--optimizer random` still runs on a machine without `gepa`.
"""

from typing import Callable, Dict

from src.optimization.optimizers.base import (
    OptimizationResult,
    PromptOptimizer,
)


def _make_gepa(**kwargs) -> PromptOptimizer:
    from src.optimization.optimizers.gepa_backend import GepaOptimizer
    return GepaOptimizer(**kwargs)


def _make_random(**kwargs) -> PromptOptimizer:
    from src.optimization.optimizers.random_search import RandomMutationOptimizer
    kwargs.pop("valset_ids", None)
    return RandomMutationOptimizer(**kwargs)


OPTIMIZERS: Dict[str, Callable[..., PromptOptimizer]] = {
    "gepa": _make_gepa,
    "random": _make_random,
}


def get_optimizer(name: str, **kwargs) -> PromptOptimizer:
    if name not in OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer {name!r}; available: {sorted(OPTIMIZERS)}"
        )
    return OPTIMIZERS[name](**kwargs)


__all__ = ["OPTIMIZERS", "get_optimizer", "OptimizationResult", "PromptOptimizer"]
