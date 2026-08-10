"""Execution-local binding for repository runtime backends."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .base import RepositoryRuntime, RepositoryRuntimeError


class RepositoryRuntimeContextError(RepositoryRuntimeError):
    """Raised when no usable runtime is bound to the current execution flow."""


_CURRENT_RUNTIME: ContextVar[RepositoryRuntime | None] = ContextVar(
    "current_repository_runtime", default=None
)


def get_repository_runtime() -> RepositoryRuntime:
    """Return the open repository runtime bound to this execution context."""

    runtime = _CURRENT_RUNTIME.get()
    if runtime is None:
        raise RepositoryRuntimeContextError(
            "No repository runtime is bound to the current execution context"
        )
    runtime.ensure_open()
    return runtime


@contextmanager
def bind_repository_runtime(runtime: RepositoryRuntime) -> Iterator[RepositoryRuntime]:
    """Temporarily bind a runtime without taking ownership of its lifecycle."""

    if not isinstance(runtime, RepositoryRuntime):
        raise TypeError("runtime must implement RepositoryRuntime")
    runtime.ensure_open()
    token = _CURRENT_RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _CURRENT_RUNTIME.reset(token)
