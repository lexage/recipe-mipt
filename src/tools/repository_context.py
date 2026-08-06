"""Runtime binding and path isolation for repository-aware tools."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.benchmarks.swe_rebench.runtime import (
    LocalRepositoryRuntime,
    bind_repository_runtime,
)


class RepositoryContextError(RuntimeError):
    """Raised when a repository tool is used without a valid runtime context."""


class RepositoryPathError(ValueError):
    """Raised when a requested path is not safe for the active repository."""


@dataclass(frozen=True)
class RepositoryContext:
    """Identify the isolated checkout used by the current benchmark task."""

    root: Path
    instance_id: str
    base_commit: str

    def __post_init__(self) -> None:
        try:
            root = Path(self.root).expanduser().resolve()
        except (OSError, RuntimeError, TypeError) as error:
            raise RepositoryContextError(
                "Repository root is not a valid path"
            ) from error
        if not root.is_dir():
            raise RepositoryContextError(
                f"Repository root does not exist or is not a directory: {root}"
            )
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise RepositoryContextError("instance_id must be a non-empty string")
        if not isinstance(self.base_commit, str) or not self.base_commit.strip():
            raise RepositoryContextError("base_commit must be a non-empty string")
        object.__setattr__(self, "root", root)

    def create_runtime(self) -> LocalRepositoryRuntime:
        """Create the local runtime used by the compatibility binding."""

        return LocalRepositoryRuntime(self.root, self.instance_id, self.base_commit)


_CURRENT_REPOSITORY: ContextVar[RepositoryContext | None] = ContextVar(
    "current_repository", default=None
)


def get_repository_context() -> RepositoryContext:
    """Return the repository bound to the current execution context."""

    context = _CURRENT_REPOSITORY.get()
    if context is None:
        raise RepositoryContextError(
            "No repository is bound to the current execution context"
        )
    return context


@contextmanager
def bind_repository_context(
    context: RepositoryContext,
) -> Iterator[RepositoryContext]:
    """Temporarily bind a task checkout for repository-aware tools."""

    if not isinstance(context, RepositoryContext):
        raise TypeError("context must be a RepositoryContext")

    token = _CURRENT_REPOSITORY.set(context)
    try:
        with bind_repository_runtime(context.create_runtime()):
            yield context
    finally:
        _CURRENT_REPOSITORY.reset(token)


def resolve_repository_path(path: str | Path = ".") -> Path:
    """Resolve a relative path while preventing access outside the checkout.

    The path need not exist, which allows future patch tools to validate paths for
    files they are about to create. Resolving the path still follows existing
    symlinks, so a symlink that points outside the checkout is rejected.
    """

    context = get_repository_context()
    requested = Path(path)

    if requested.is_absolute():
        raise RepositoryPathError(f"Absolute paths are not allowed: {path}")
    if ".git" in requested.parts:
        raise RepositoryPathError("Access to .git is not allowed")

    try:
        resolved = (context.root / requested).resolve()
    except (OSError, RuntimeError) as error:
        raise RepositoryPathError(
            f"Could not resolve repository path: {path}"
        ) from error

    if not resolved.is_relative_to(context.root):
        raise RepositoryPathError(f"Path is outside the repository: {path}")

    return resolved
