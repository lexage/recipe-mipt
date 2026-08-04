"""Backend-neutral runtime interface for SWE-rebench repository operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class RepositoryRuntimeError(RuntimeError):
    """Raised when a repository runtime cannot perform an operation."""


class RepositoryRuntimeClosedError(RepositoryRuntimeError):
    """Raised when an operation is requested after a runtime was closed."""


@dataclass(frozen=True)
class CommandResult:
    """Normalized result of a command executed by any repository backend."""

    command: str
    exit_code: Optional[int]
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("command must be a non-empty string")
        if self.exit_code is not None and not isinstance(self.exit_code, int):
            raise ValueError("exit_code must be an integer or None")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValueError("stdout and stderr must be strings")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if self.timed_out and self.exit_code is not None:
            raise ValueError("a timed out command cannot have an exit code")


class RepositoryRuntime(ABC):
    """Access a task repository without exposing its execution backend to tools.

    Implementations may operate on a host checkout, a Docker container, or
    another isolated environment. Public repository tools should depend only on
    this interface.
    """

    def __init__(
        self,
        instance_id: str,
        base_commit: str,
        workdir: str,
    ) -> None:
        for name, value in (
            ("instance_id", instance_id),
            ("base_commit", base_commit),
            ("workdir", workdir),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        self.instance_id = instance_id
        self.base_commit = base_commit
        self.workdir = workdir
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    def ensure_open(self) -> None:
        if self._closed:
            raise RepositoryRuntimeClosedError(
                f"Repository runtime for {self.instance_id} is closed"
            )

    def close(self) -> None:
        """Release runtime resources; repeated calls are safe."""

        self._closed = True

    def __enter__(self) -> "RepositoryRuntime":
        self.ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @abstractmethod
    def list_files(
        self,
        path: str = ".",
        *,
        max_depth: int = 3,
        max_entries: int = 200,
        include_hidden: bool = False,
    ) -> str:
        """Return a bounded listing rooted at a repository-relative path."""

    @abstractmethod
    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int = 200,
        max_lines: int = 400,
        max_file_bytes: int = 2_000_000,
    ) -> str:
        """Return a bounded, numbered range from a repository text file."""

    @abstractmethod
    def search_code(
        self,
        query: str,
        *,
        path: str = ".",
        glob: Optional[str] = None,
        regex: bool = False,
        max_results: int = 100,
        timeout: int = 30,
        max_output_chars: int = 30_000,
    ) -> str:
        """Search repository contents and return bounded textual matches."""

    @abstractmethod
    def apply_patch(
        self,
        patch: str,
        *,
        timeout: int = 30,
        max_patch_bytes: int = 1_000_000,
    ) -> str:
        """Validate and apply a unified diff to the task repository."""

    @abstractmethod
    def run_command(
        self,
        command: str,
        *,
        timeout: int = 120,
        max_output_chars: int = 30_000,
    ) -> CommandResult:
        """Execute a bounded, non-interactive command in the task repository."""

    @abstractmethod
    def get_diff(
        self,
        path: str = ".",
        *,
        stat_only: bool = False,
        timeout: int = 30,
        max_output_chars: int = 50_000,
    ) -> str:
        """Return status and changes relative to the task base commit."""
