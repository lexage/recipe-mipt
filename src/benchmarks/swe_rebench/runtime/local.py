"""Host-checkout implementation of the SWE-rebench repository runtime."""

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

from .base import CommandResult, RepositoryRuntime, RepositoryRuntimeError


class LocalRepositoryRuntime(RepositoryRuntime):
    """Operate on an isolated repository checkout on the local filesystem."""

    def __init__(self, root: str | Path, instance_id: str, base_commit: str) -> None:
        try:
            resolved_root = Path(root).expanduser().resolve()
        except (OSError, RuntimeError, TypeError) as error:
            raise RepositoryRuntimeError(
                "Repository root is not a valid path"
            ) from error
        if not resolved_root.is_dir():
            raise RepositoryRuntimeError(
                f"Repository root does not exist or is not a directory: {resolved_root}"
            )
        self.root = resolved_root
        super().__init__(instance_id, base_commit, str(resolved_root))

    def resolve_path(self, path: str | Path = ".") -> Path:
        """Resolve a relative path without allowing checkout boundary escapes."""

        self.ensure_open()
        requested = Path(path)
        if requested.is_absolute():
            raise RepositoryRuntimeError(f"Absolute paths are not allowed: {path}")
        if ".git" in requested.parts:
            raise RepositoryRuntimeError("Access to .git is not allowed")
        try:
            resolved = (self.root / requested).resolve()
        except (OSError, RuntimeError) as error:
            raise RepositoryRuntimeError(
                f"Could not resolve repository path: {path}"
            ) from error
        if not resolved.is_relative_to(self.root):
            raise RepositoryRuntimeError(f"Path is outside the repository: {path}")
        return resolved

    def list_files(
        self,
        path: str = ".",
        *,
        max_depth: int = 3,
        max_entries: int = 200,
        include_hidden: bool = False,
    ) -> str:
        self._require_positive(max_entries=max_entries)
        if max_depth < 0:
            raise RepositoryRuntimeError("max_depth must be non-negative")
        directory = self.resolve_path(path)
        if not directory.exists():
            return f"Path does not exist: {path}"
        if not directory.is_dir():
            return f"Path is not a directory: {path}"

        entries: list[str] = []
        truncated = False
        for candidate in sorted(directory.rglob("*")):
            relative = candidate.relative_to(directory)
            if len(relative.parts) > max_depth:
                continue
            if not include_hidden and any(
                part.startswith(".") for part in relative.parts
            ):
                continue
            suffix = "/" if candidate.is_dir() else ""
            entries.append(f"{relative.as_posix()}{suffix}")
            if len(entries) == max_entries:
                truncated = True
                break

        header = f"Directory: {Path(path).as_posix()}"
        body = "\n".join(entries) if entries else "[empty]"
        footer = f"\n\n[{len(entries)} entries"
        if truncated:
            footer += ", output truncated"
        return f"{header}\n\n{body}{footer}]"

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int = 200,
        max_lines: int = 400,
        max_file_bytes: int = 2_000_000,
    ) -> str:
        self._require_positive(max_lines=max_lines, max_file_bytes=max_file_bytes)
        if start_line < 1 or end_line < start_line:
            return "Invalid line range: require 1 <= start_line <= end_line"
        if end_line - start_line + 1 > max_lines:
            end_line = start_line + max_lines - 1

        file_path = self.resolve_path(path)
        if not file_path.is_file():
            return f"File does not exist: {path}"
        if file_path.stat().st_size > max_file_bytes:
            return f"File is too large to read: {path}"
        data = file_path.read_bytes()
        if b"\x00" in data:
            return f"Binary file cannot be read: {path}"
        try:
            lines = data.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            return f"File is not valid UTF-8 text: {path}"

        selected = lines[start_line - 1 : end_line]
        rendered = "\n".join(
            f"{number:>6} | {line}"
            for number, line in enumerate(selected, start=start_line)
        )
        actual_end = start_line + len(selected) - 1
        if not selected:
            actual_end = start_line - 1
            rendered = "[no lines in requested range]"
        return (
            f"File: {path}\nLines: {start_line}-{actual_end} of {len(lines)}\n\n"
            f"{rendered}"
        )

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
        self._require_positive(
            max_results=max_results,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )
        if not query:
            return "Search query must not be empty"
        search_root = self.resolve_path(path)
        if not search_root.exists():
            return f"Search path does not exist: {path}"
        relative_root = search_root.relative_to(self.root)
        pathspec = "." if not relative_root.parts else relative_root.as_posix()
        command = ["rg", "--line-number", "--column", "--color", "never"]
        if not regex:
            command.append("--fixed-strings")
        if glob:
            command.extend(["--glob", glob])
        command.extend(["--glob", "!.git/**", "--", query, pathspec])

        try:
            result = subprocess.run(
                command,
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return "Search failed: ripgrep (rg) is not installed"
        except subprocess.TimeoutExpired:
            return f"Search timed out after {timeout} seconds"
        if result.returncode == 1:
            return "No matches found"
        if result.returncode != 0:
            return f"Search failed (exit {result.returncode}): {result.stderr.strip()}"

        lines = result.stdout.splitlines()
        truncated = len(lines) > max_results
        output = "\n".join(lines[:max_results])
        if len(output) > max_output_chars:
            output = output[:max_output_chars]
            truncated = True
        if truncated:
            output += "\n[output truncated]"
        return output

    def apply_patch(
        self,
        patch: str,
        *,
        timeout: int = 30,
        max_patch_bytes: int = 1_000_000,
    ) -> str:
        self.ensure_open()
        self._require_positive(timeout=timeout, max_patch_bytes=max_patch_bytes)
        if not patch.strip():
            return "Patch must not be empty"
        if len(patch.encode("utf-8")) > max_patch_bytes:
            return f"Patch exceeds the {max_patch_bytes}-byte limit"
        try:
            paths = self._validate_patch_paths(patch)
        except RepositoryRuntimeError as error:
            return f"Patch rejected: {error}"

        try:
            check = subprocess.run(
                ["git", "apply", "--check", "-"],
                cwd=self.root,
                input=patch,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            if check.returncode != 0:
                detail = check.stderr.strip() or check.stdout.strip()
                return f"Patch check failed: {detail}"
            applied = subprocess.run(
                ["git", "apply", "-"],
                cwd=self.root,
                input=patch,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return "Patch failed: git is not installed"
        except subprocess.TimeoutExpired:
            return f"Patch command timed out after {timeout} seconds"
        if applied.returncode != 0:
            detail = applied.stderr.strip() or applied.stdout.strip()
            return f"Patch apply failed: {detail}"
        return "Patch applied successfully.\n\nChanged paths:\n" + "\n".join(
            f"- {path}" for path in paths
        )

    def run_command(
        self,
        command: str,
        *,
        timeout: int = 120,
        max_output_chars: int = 30_000,
    ) -> CommandResult:
        self.ensure_open()
        self._require_positive(timeout=timeout, max_output_chars=max_output_chars)
        try:
            arguments = shlex.split(command)
        except ValueError as error:
            raise RepositoryRuntimeError(f"Invalid command: {error}") from error
        if not arguments:
            raise RepositoryRuntimeError("Command must not be empty")

        started = time.monotonic()
        try:
            result = subprocess.run(
                arguments,
                cwd=self.root,
                env=self._environment(),
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as error:
            raise RepositoryRuntimeError(
                f"Command not found: {arguments[0]}"
            ) from error
        except subprocess.TimeoutExpired as error:
            return CommandResult(
                command=command,
                exit_code=None,
                stdout=self._as_text(error.stdout, max_output_chars),
                stderr=self._as_text(error.stderr, max_output_chars),
                duration_seconds=time.monotonic() - started,
                timed_out=True,
            )
        return CommandResult(
            command=command,
            exit_code=result.returncode,
            stdout=result.stdout[:max_output_chars],
            stderr=result.stderr[:max_output_chars],
            duration_seconds=time.monotonic() - started,
        )

    def get_diff(
        self,
        path: str = ".",
        *,
        stat_only: bool = False,
        timeout: int = 30,
        max_output_chars: int = 50_000,
    ) -> str:
        self._require_positive(timeout=timeout, max_output_chars=max_output_chars)
        selected = self.resolve_path(path)
        relative = selected.relative_to(self.root)
        pathspec = "." if not relative.parts else relative.as_posix()
        try:
            intent_to_add = self._git(
                ["add", "--intent-to-add", "--", pathspec], timeout
            )
            status = self._git(["status", "--short", "--", pathspec], timeout)
            diff_args = ["diff", "--binary"]
            if stat_only:
                diff_args.append("--stat")
            diff_args.extend([self.base_commit, "--", pathspec])
            diff = self._git(diff_args, timeout)
        except FileNotFoundError:
            return "Diff failed: git is not installed"
        except subprocess.TimeoutExpired:
            return f"Diff command timed out after {timeout} seconds"

        if any(result.returncode != 0 for result in (intent_to_add, status, diff)):
            detail = (
                intent_to_add.stderr.strip()
                or status.stderr.strip()
                or diff.stderr.strip()
            )
            return f"Diff failed: {detail}"
        status_text = status.stdout.strip() or "[clean]"
        diff_text = diff.stdout.strip() or "[no diff]"
        output = f"Status:\n{status_text}\n\nDiff:\n{diff_text}"
        if len(output) > max_output_chars:
            output = output[:max_output_chars] + "\n[output truncated]"
        return output

    def _validate_patch_paths(self, patch: str) -> list[str]:
        paths: list[str] = []
        for line in patch.splitlines():
            if not line.startswith("diff --git "):
                continue
            try:
                parts = shlex.split(line)
            except ValueError as error:
                raise RepositoryRuntimeError("Malformed diff header") from error
            if len(parts) != 4:
                raise RepositoryRuntimeError("Malformed diff header")
            for raw_path, prefix in ((parts[2], "a/"), (parts[3], "b/")):
                if not raw_path.startswith(prefix):
                    raise RepositoryRuntimeError(f"Unexpected diff path: {raw_path}")
                relative = raw_path[len(prefix) :]
                self.resolve_path(relative)
                paths.append(relative)
        if not paths:
            raise RepositoryRuntimeError("Patch has no 'diff --git' file headers")
        return sorted(set(paths))

    def _git(self, arguments: list[str], timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def _environment(self) -> dict[str, str]:
        allowed = ("PATH", "LANG", "LC_ALL", "TERM", "PYTHONPATH")
        environment = {key: os.environ[key] for key in allowed if key in os.environ}
        environment.update({"HOME": str(self.root), "GIT_TERMINAL_PROMPT": "0"})
        return environment

    @staticmethod
    def _as_text(output: str | bytes | None, limit: int) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return output[:limit]

    @staticmethod
    def _require_positive(**values: int) -> None:
        invalid = [name for name, value in values.items() if value < 1]
        if invalid:
            raise RepositoryRuntimeError(
                "Runtime limits must be positive: " + ", ".join(invalid)
            )
