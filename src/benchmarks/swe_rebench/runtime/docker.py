"""Docker container lifecycle for SWE-rebench inference runtimes."""

import io
import re
import shlex
import tarfile
import time
import uuid
from typing import Any, Literal, Optional

from ..images import InstanceImage
from .base import CommandResult, RepositoryRuntime, RepositoryRuntimeError

PullPolicy = Literal["always", "missing", "never"]


class DockerRepositoryRuntime(RepositoryRuntime):
    """Own a dedicated inference container created from an instance image.

    Repository operations are deliberately completed in a later layer; this
    class owns image selection, container startup, checkout validation, command
    execution and reliable cleanup.
    """

    def __init__(
        self,
        *,
        client: Any,
        container: Any,
        image: InstanceImage,
        instance_id: str,
        base_commit: str,
        keep_container: bool = False,
    ) -> None:
        if client is None or container is None:
            raise RepositoryRuntimeError("Docker client and container are required")
        if not isinstance(image, InstanceImage):
            raise TypeError("image must be an InstanceImage")
        self.client = client
        self.container = container
        self.image = image
        self.keep_container = keep_container
        self.cleanup_error: Optional[str] = None
        super().__init__(instance_id, base_commit, image.workdir)

    @classmethod
    def create(
        cls,
        *,
        client: Any,
        image: InstanceImage,
        instance_id: str,
        base_commit: str,
        run_id: str,
        pull_policy: PullPolicy = "missing",
        keep_container: bool = False,
        network_disabled: bool = True,
        memory_limit: Optional[str] = None,
        nano_cpus: Optional[int] = None,
        container_name: Optional[str] = None,
    ) -> "DockerRepositoryRuntime":
        """Pull if needed, start one container, and validate its clean checkout."""

        if client is None:
            raise RepositoryRuntimeError("Docker client is required")
        if not isinstance(image, InstanceImage):
            raise TypeError("image must be an InstanceImage")
        if pull_policy not in {"always", "missing", "never"}:
            raise RepositoryRuntimeError(f"Unknown pull policy: {pull_policy}")
        if not isinstance(run_id, str) or not run_id.strip():
            raise RepositoryRuntimeError("run_id must be a non-empty string")
        if nano_cpus is not None and nano_cpus < 1:
            raise RepositoryRuntimeError("nano_cpus must be positive")

        cls._ensure_image(client, image.name, pull_policy)
        name = container_name or cls._container_name(instance_id, run_id)
        labels = {
            "swe-rebench.role": "inference",
            "swe-rebench.instance_id": instance_id,
            "swe-rebench.run_id": run_id,
        }
        options: dict[str, Any] = {
            "image": image.name,
            "command": ["sleep", "infinity"],
            "name": name,
            "detach": True,
            "stdin_open": False,
            "tty": False,
            "working_dir": image.workdir,
            "platform": image.platform,
            "network_disabled": network_disabled,
            "labels": labels,
        }
        if memory_limit is not None:
            options["mem_limit"] = memory_limit
        if nano_cpus is not None:
            options["nano_cpus"] = nano_cpus

        container = None
        runtime = None
        try:
            container = client.containers.run(**options)
            runtime = cls(
                client=client,
                container=container,
                image=image,
                instance_id=instance_id,
                base_commit=base_commit,
                keep_container=keep_container,
            )
            runtime.validate_checkout()
            return runtime
        except Exception as error:
            if runtime is not None:
                runtime.close()
            elif container is not None:
                cls._discard_container(container)
            if isinstance(error, RepositoryRuntimeError):
                raise
            raise RepositoryRuntimeError(
                f"Could not start inference container for {instance_id}: {error}"
            ) from error

    def validate_checkout(self) -> None:
        """Ensure /testbed is a clean Git checkout at the requested base commit."""

        self.ensure_open()
        checks = (
            ("test -d .git", "Task workdir is not a Git checkout"),
            ("git rev-parse HEAD", "Could not determine task checkout commit"),
            ("git status --porcelain", "Could not inspect task checkout status"),
        )
        results: list[CommandResult] = []
        for command, message in checks:
            result = self.run_command(command)
            if result.exit_code != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RepositoryRuntimeError(f"{message}: {detail}")
            results.append(result)

        head = results[1].stdout.strip()
        if head != self.base_commit:
            raise RepositoryRuntimeError(
                f"Task checkout is at {head}, expected {self.base_commit}"
            )
        status = results[2].stdout.strip()
        if status:
            raise RepositoryRuntimeError(
                "Task checkout is not clean before inference:\n" + status
            )

    def run_command(
        self,
        command: str,
        *,
        timeout: int = 120,
        max_output_chars: int = 30_000,
    ) -> CommandResult:
        """Execute a non-interactive command in the owned container.

        Commands are wrapped by GNU ``timeout`` inside the instance container,
        so a timed-out tool call cannot keep running after control returns.
        """

        self.ensure_open()
        if not isinstance(command, str) or not command.strip():
            raise RepositoryRuntimeError("Command must be a non-empty string")
        if timeout < 1 or max_output_chars < 1:
            raise RepositoryRuntimeError("Command limits must be positive")

        started = time.monotonic()
        try:
            result = self.container.exec_run(
                [
                    "timeout",
                    "--signal=KILL",
                    f"{timeout}s",
                    "/bin/sh",
                    "-lc",
                    command,
                ],
                workdir=self.workdir,
                stdin=False,
                tty=False,
                demux=True,
            )
        except Exception as error:
            raise RepositoryRuntimeError(
                f"Container command failed to start: {error}"
            ) from error
        exit_code, stdout, stderr = self._unpack_exec_result(result)
        timed_out = exit_code in {124, 137}
        return CommandResult(
            command=command,
            exit_code=None if timed_out else exit_code,
            stdout=self._decode(stdout)[:max_output_chars],
            stderr=self._decode(stderr)[:max_output_chars],
            duration_seconds=time.monotonic() - started,
            timed_out=timed_out,
        )

    def close(self) -> None:
        """Stop and remove the owned container unless debug retention is enabled."""

        if self.is_closed:
            return
        errors: list[str] = []
        try:
            if not self.keep_container:
                try:
                    self.container.stop(timeout=10)
                except Exception as error:
                    errors.append(f"stop failed: {error}")
                try:
                    self.container.remove(force=True)
                except Exception as error:
                    errors.append(f"remove failed: {error}")
        finally:
            super().close()
        if errors:
            self.cleanup_error = "; ".join(errors)

    def list_files(
        self,
        path: str = ".",
        *,
        max_depth: int = 3,
        max_entries: int = 200,
        include_hidden: bool = False,
    ) -> str:
        if max_depth < 0 or max_entries < 1:
            raise RepositoryRuntimeError("Invalid file listing limits")
        relative = self._resolve_path(path)
        quoted = shlex.quote(relative)
        hidden = "" if include_hidden else " -not -path '*/.*'"
        command = (
            f"test -d {quoted} || exit 3; "
            f"find {quoted} -mindepth 1 -maxdepth {max_depth}{hidden} "
            f"-printf '%P\t%y\n' | LC_ALL=C sort | head -n {max_entries + 1}"
        )
        result = self.run_command(command)
        if result.exit_code == 3:
            return f"Path is not a directory: {path}"
        self._require_success(result, "Could not list repository files")
        raw_entries = result.stdout.splitlines()
        truncated = len(raw_entries) > max_entries
        entries = []
        for line in raw_entries[:max_entries]:
            name, _, kind = line.rpartition("\t")
            entries.append(name + ("/" if kind == "d" else ""))
        body = "\n".join(entries) if entries else "[empty]"
        footer = f"\n\n[{len(entries)} entries"
        if truncated:
            footer += ", output truncated"
        return f"Directory: {path}\n\n{body}{footer}]"

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int = 200,
        max_lines: int = 400,
        max_file_bytes: int = 2_000_000,
    ) -> str:
        if min(max_lines, max_file_bytes) < 1:
            raise RepositoryRuntimeError("Invalid file read limits")
        if start_line < 1 or end_line < start_line:
            return "Invalid line range: require 1 <= start_line <= end_line"
        end_line = min(end_line, start_line + max_lines - 1)
        relative = self._resolve_path(path)
        quoted = shlex.quote(relative)
        metadata = self.run_command(
            f"test -f {quoted} || exit 3; "
            f"size=$(wc -c < {quoted}); lines=$(awk 'END {{print NR}}' {quoted}); "
            'printf \'%s\\n%s\\n\' "$size" "$lines"'
        )
        if metadata.exit_code == 3:
            return f"File does not exist: {path}"
        self._require_success(metadata, "Could not inspect repository file")
        try:
            size_text, line_count_text = metadata.stdout.splitlines()[:2]
            size, line_count = int(size_text), int(line_count_text)
        except (ValueError, IndexError) as error:
            raise RepositoryRuntimeError(
                "Invalid file metadata from container"
            ) from error
        if size > max_file_bytes:
            return f"File is too large to read: {path}"
        binary = self.run_command(
            f"test ! -s {quoted} || LC_ALL=C grep -Iq '' {quoted}", timeout=30
        )
        if binary.exit_code not in {0}:
            return f"Binary file cannot be read: {path}"
        content = self.run_command(f"sed -n '{start_line},{end_line}p' {quoted}")
        self._require_success(content, "Could not read repository file")
        selected = content.stdout.splitlines()
        rendered = "\n".join(
            f"{number:>6} | {line}"
            for number, line in enumerate(selected, start=start_line)
        )
        actual_end = start_line + len(selected) - 1
        if not selected:
            actual_end = start_line - 1
            rendered = "[no lines in requested range]"
        return (
            f"File: {path}\nLines: {start_line}-{actual_end} of {line_count}\n\n"
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
        if not query:
            return "Search query must not be empty"
        if min(max_results, timeout, max_output_chars) < 1:
            raise RepositoryRuntimeError("Invalid search limits")
        relative = self._resolve_path(path)
        fixed = "" if regex else " --fixed-strings"
        glob_arg = f" --glob {shlex.quote(glob)}" if glob else ""
        rg = (
            "rg --line-number --column --color never"
            f"{fixed}{glob_arg} --glob '!.git/**' -- "
            f"{shlex.quote(query)} {shlex.quote(relative)}"
        )
        grep_fixed = "" if regex else " -F"
        if glob:
            grep_path = glob if relative == "." else f"{relative.rstrip('/')}/{glob}"
        else:
            grep_path = relative
        git_grep = (
            f"git grep -n{grep_fixed} -- {shlex.quote(query)} -- "
            f"{shlex.quote(grep_path)}"
        )
        command = f"if command -v rg >/dev/null 2>&1; then {rg}; else {git_grep}; fi"
        result = self.run_command(
            command, timeout=timeout, max_output_chars=max_output_chars
        )
        if result.timed_out:
            return f"Search timed out after {timeout} seconds"
        if result.exit_code == 1:
            return "No matches found"
        self._require_success(result, "Search failed")
        lines = result.stdout.splitlines()
        truncated = len(lines) > max_results
        output = "\n".join(lines[:max_results])[:max_output_chars]
        if truncated or len(result.stdout) > max_output_chars:
            output += "\n[output truncated]"
        return output

    def apply_patch(
        self,
        patch: str,
        *,
        timeout: int = 30,
        max_patch_bytes: int = 1_000_000,
    ) -> str:
        if not patch.strip():
            return "Patch must not be empty"
        if len(patch.encode("utf-8")) > max_patch_bytes:
            return f"Patch exceeds the {max_patch_bytes}-byte limit"
        paths = self._validate_patch_paths(patch)
        patch_name = f"agent-{uuid.uuid4().hex}.patch"
        archive = self._patch_archive(patch_name, patch)
        try:
            self.container.put_archive("/tmp", archive)
        except Exception as error:
            raise RepositoryRuntimeError(
                f"Could not copy patch to container: {error}"
            ) from error
        patch_path = f"/tmp/{patch_name}"
        try:
            check = self.run_command(
                f"git apply --check {shlex.quote(patch_path)}", timeout=timeout
            )
            if check.exit_code != 0:
                detail = check.stderr.strip() or check.stdout.strip()
                return f"Patch check failed: {detail}"
            applied = self.run_command(
                f"git apply {shlex.quote(patch_path)}", timeout=timeout
            )
            self._require_success(applied, "Patch apply failed")
        finally:
            self.run_command(f"rm -f {shlex.quote(patch_path)}", timeout=30)
        return "Patch applied successfully.\n\nChanged paths:\n" + "\n".join(
            f"- {path}" for path in paths
        )

    def get_diff(
        self,
        path: str = ".",
        *,
        stat_only: bool = False,
        timeout: int = 30,
        max_output_chars: int = 50_000,
    ) -> str:
        relative = self._resolve_path(path)
        quoted = shlex.quote(relative)
        intent = self.run_command(
            f"git add --intent-to-add -- {quoted}", timeout=timeout
        )
        self._require_success(intent, "Diff failed")
        status = self.run_command(f"git status --short -- {quoted}", timeout=timeout)
        self._require_success(status, "Diff failed")
        stat = " --stat" if stat_only else ""
        diff = self.run_command(
            f"git diff --binary{stat} {shlex.quote(self.base_commit)} -- {quoted}",
            timeout=timeout,
            max_output_chars=max_output_chars,
        )
        self._require_success(diff, "Diff failed")
        status_text = status.stdout.strip() or "[clean]"
        diff_text = diff.stdout.strip() or "[no diff]"
        output = f"Status:\n{status_text}\n\nDiff:\n{diff_text}"
        if len(output) > max_output_chars:
            output = output[:max_output_chars] + "\n[output truncated]"
        return output

    def _resolve_path(self, path: str) -> str:
        self.ensure_open()
        if not isinstance(path, str) or not path:
            raise RepositoryRuntimeError("Repository path must be a non-empty string")
        if path.startswith("/"):
            raise RepositoryRuntimeError(f"Absolute paths are not allowed: {path}")
        parts = [part for part in path.split("/") if part not in {"", "."}]
        if ".." in parts:
            raise RepositoryRuntimeError(f"Path is outside the repository: {path}")
        if ".git" in parts:
            raise RepositoryRuntimeError("Access to .git is not allowed")
        normalized = "/".join(parts) or "."
        quoted = shlex.quote(normalized)
        workdir = shlex.quote(self.workdir)
        check = self.run_command(
            f"resolved=$(realpath -m -- {quoted}) || exit 2; "
            f'case "$resolved" in {workdir}|{workdir}/*) exit 0;; *) exit 3;; esac',
            timeout=30,
        )
        if check.exit_code == 3:
            raise RepositoryRuntimeError(f"Path is outside the repository: {path}")
        self._require_success(check, "Could not validate repository path")
        return normalized

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
                self._resolve_path(relative)
                paths.append(relative)
        if not paths:
            raise RepositoryRuntimeError("Patch has no 'diff --git' file headers")
        return sorted(set(paths))

    @staticmethod
    def _patch_archive(name: str, patch: str) -> bytes:
        payload = patch.encode("utf-8")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(payload))
        return archive.getvalue()

    @staticmethod
    def _require_success(result: CommandResult, message: str) -> None:
        if result.timed_out:
            raise RepositoryRuntimeError(f"{message}: command timed out")
        if result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RepositoryRuntimeError(f"{message}: {detail}")

    @staticmethod
    def _ensure_image(client: Any, image_name: str, pull_policy: PullPolicy) -> None:
        if pull_policy == "always":
            client.images.pull(image_name)
            return
        try:
            client.images.get(image_name)
        except Exception as error:
            if pull_policy == "never":
                raise RepositoryRuntimeError(
                    f"Docker image is not available locally: {image_name}"
                ) from error
            client.images.pull(image_name)

    @staticmethod
    def _container_name(instance_id: str, run_id: str) -> str:
        raw = f"swe-rebench-infer-{run_id}-{instance_id}-{uuid.uuid4().hex[:8]}"
        normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw).strip("-_.")
        if not normalized:
            raise RepositoryRuntimeError("Could not create a Docker container name")
        return normalized[:128]

    @staticmethod
    def _discard_container(container: Any) -> None:
        try:
            container.remove(force=True)
        except Exception:
            pass

    @staticmethod
    def _unpack_exec_result(result: Any) -> tuple[int, Any, Any]:
        exit_code = getattr(result, "exit_code", None)
        output = getattr(result, "output", None)
        if exit_code is None and isinstance(result, tuple) and len(result) == 2:
            exit_code, output = result
        if not isinstance(exit_code, int):
            raise RepositoryRuntimeError("Docker exec returned no integer exit code")
        if isinstance(output, tuple) and len(output) == 2:
            return exit_code, output[0], output[1]
        return exit_code, output, b""

    @staticmethod
    def _decode(output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return str(output)
