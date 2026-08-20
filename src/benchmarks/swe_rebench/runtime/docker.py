"""Docker container lifecycle for SWE-rebench inference runtimes."""

import difflib
import io
import logging
import posixpath
import re
import shlex
import tarfile
import time
import uuid
from typing import Any, Literal, Optional

from ..images import InstanceImage
from .base import (
    CommandResult,
    FileEditError,
    RepositoryRuntime,
    RepositoryRuntimeError,
    TextReplaceError,
)

PullPolicy = Literal["always", "missing", "never"]

COMMAND_ENVIRONMENT = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

CONDA_ACTIVATION = """
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh || exit 127
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh || exit 127
elif [ -f /opt/miniconda3/bin/activate ]; then
    source /opt/miniconda3/bin/activate || exit 127
else
    echo "Conda activation script not found" >&2
    exit 127
fi
conda activate testbed || exit 127
""".strip()


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
        logging.info(
            "STAGE\tDOCKER_IMAGE_READY\tinstance_id=%s\timage=%s\tpull_policy=%s",
            instance_id,
            image.name,
            pull_policy,
        )
        name = container_name or cls._container_name(instance_id, run_id)
        labels = {
            "swe-rebench.role": "inference",
            "swe-rebench.instance_id": instance_id,
            "swe-rebench.run_id": run_id,
        }
        options: dict[str, Any] = {
            "image": image.name,
            "command": ["tail", "-f", "/dev/null"],
            "name": name,
            "detach": True,
            "stdin_open": False,
            "tty": False,
            "working_dir": image.workdir,
            "platform": image.platform,
            "network_disabled": network_disabled,
            "labels": labels,
            "user": image.user,
            "cap_add": list(image.cap_add),
            "environment": COMMAND_ENVIRONMENT,
        }
        if memory_limit is not None:
            options["mem_limit"] = memory_limit
        if nano_cpus is not None:
            options["nano_cpus"] = nano_cpus

        container = None
        runtime = None
        try:
            container = client.containers.run(**options)
            logging.info(
                "STAGE\tDOCKER_CONTAINER_CREATED\tinstance_id=%s\tcontainer_id=%s\tcontainer_name=%s",
                instance_id,
                getattr(container, "id", "unknown"),
                getattr(container, "name", name),
            )
            runtime = cls(
                client=client,
                container=container,
                image=image,
                instance_id=instance_id,
                base_commit=base_commit,
                keep_container=keep_container,
            )
            runtime.validate_checkout()
            logging.info(
                "STAGE\tDOCKER_CHECKOUT_VALIDATED\tinstance_id=%s\tbase_commit=%s\tworkdir=%s",
                instance_id,
                base_commit,
                image.workdir,
            )
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
            ("pwd", "Could not determine task working directory"),
            ("test -d .git", "Task workdir is not a Git checkout"),
            ("git rev-parse HEAD", "Could not determine task checkout commit"),
            ("git status --porcelain", "Could not inspect task checkout status"),
            ("command -v python", "Task environment has no Python executable"),
            ("python --version", "Could not inspect task Python version"),
            (
                'test "${CONDA_DEFAULT_ENV:-}" = "testbed"',
                "Task conda environment 'testbed' is not active",
            ),
            (
                "python -c \"import sys; print(sys.prefix); "
                "assert sys.prefix.endswith('/envs/testbed'), sys.prefix\"",
                "Task Python is not from the 'testbed' environment",
            ),
        )
        results: list[CommandResult] = []
        for command, message in checks:
            result = self.run_command(command)
            if result.exit_code != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RepositoryRuntimeError(f"{message}: {detail}")
            results.append(result)

        head = results[2].stdout.strip()
        if head != self.base_commit:
            raise RepositoryRuntimeError(
                f"Task checkout is at {head}, expected {self.base_commit}"
            )
        status = results[3].stdout.strip()
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

        wrapped_command = self._wrap_command(command)
        started = time.monotonic()
        try:
            result = self.container.exec_run(
                [
                    "timeout",
                    "--signal=KILL",
                    f"{timeout}s",
                    "/bin/bash",
                    "-c",
                    wrapped_command,
                ],
                workdir=self.workdir,
                environment=COMMAND_ENVIRONMENT,
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

    @staticmethod
    def _wrap_command(command: str) -> str:
        """Activate the task environment before every isolated Docker exec."""

        return f"{CONDA_ACTIVATION}\n{command}"

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
                if not errors:
                    logging.info(
                        "STAGE\tDOCKER_CONTAINER_REMOVED\tinstance_id=%s\tcontainer_id=%s",
                        self.instance_id,
                        getattr(self.container, "id", "unknown"),
                    )
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
            return f"Path is not a directory: {relative}"
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
        return f"Directory: {relative}\n\n{body}{footer}]"

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int = 200,
        max_lines: int = 400,
        max_file_bytes: int = 2_000_000,
        line_numbers: bool = False,
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
            return f"File does not exist: {relative}"
        self._require_success(metadata, "Could not inspect repository file")
        try:
            size_text, line_count_text = metadata.stdout.splitlines()[:2]
            size, line_count = int(size_text), int(line_count_text)
        except (ValueError, IndexError) as error:
            raise RepositoryRuntimeError(
                "Invalid file metadata from container"
            ) from error
        if size > max_file_bytes:
            return f"File is too large to read: {relative}"
        binary = self.run_command(
            f"test ! -s {quoted} || LC_ALL=C grep -Iq '' {quoted}", timeout=30
        )
        if binary.exit_code not in {0}:
            return f"Binary file cannot be read: {relative}"
        content = self.run_command(f"sed -n '{start_line},{end_line}p' {quoted}")
        self._require_success(content, "Could not read repository file")
        if not line_numbers:
            return content.stdout.removesuffix("\n")
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
            f"File: {relative}\nLines: {start_line}-{actual_end} of {line_count}\n\n"
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

    def apply_file_edit(
        self,
        operation: str,
        path: str,
        *,
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        destination: Optional[str] = None,
        max_file_bytes: int = 2_000_000,
    ) -> str:
        if (
            not isinstance(max_file_bytes, int)
            or isinstance(max_file_bytes, bool)
            or max_file_bytes < 1
        ):
            raise FileEditError(
                "max_file_bytes must be a positive integer",
                error_code="invalid_edit_limits",
                retryable=False,
            )
        if operation == "replace":
            if old_text is None or new_text is None:
                raise FileEditError(
                    "replace requires old_text and new_text",
                    error_code="invalid_edit_arguments",
                    retryable=False,
                )
            result = self.replace_text(
                path,
                old_text,
                new_text,
                expected_replacements=1,
                max_file_bytes=max_file_bytes,
            )
            changed_paths = self._verify_file_edit(path)
            return f"{result}\nChanged paths: {', '.join(changed_paths)}"

        if operation == "create":
            if not isinstance(new_text, str):
                raise FileEditError(
                    "create requires new_text",
                    error_code="invalid_edit_arguments",
                    retryable=False,
                )
            payload = new_text.encode("utf-8")
            if len(payload) > max_file_bytes:
                raise FileEditError(
                    f"New file exceeds the {max_file_bytes}-byte limit",
                    error_code="file_too_large",
                    retryable=False,
                )
            relative = self._resolve_path(path)
            self._write_text_atomically(relative, payload, refuse_existing=True)
            changed_paths = self._verify_file_edit(relative)
            return (
                f"Created {relative}. Changed paths: {', '.join(changed_paths)}. "
                "Review the repository changes."
            )

        if operation == "delete":
            relative = self._resolve_path(path)
            quoted = shlex.quote(relative)
            result = self.run_command(
                f"test -f {quoted} || exit 17; rm -- {quoted}",
                timeout=30,
            )
            if result.exit_code == 17:
                raise FileEditError(
                    f"File does not exist or is not a regular file: {relative}",
                    error_code="file_not_found",
                )
            self._require_success(result, f"Could not delete {relative}")
            changed_paths = self._verify_file_edit(relative)
            return (
                f"Deleted {relative}. Changed paths: {', '.join(changed_paths)}. "
                "Review the repository changes."
            )

        if operation == "move":
            if not isinstance(destination, str) or not destination:
                raise FileEditError(
                    "move requires destination",
                    error_code="invalid_edit_arguments",
                    retryable=False,
                )
            source = self._resolve_path(path)
            target = self._resolve_path(destination)
            if source == target:
                raise FileEditError(
                    "Move destination must differ from path",
                    error_code="no_change",
                    retryable=False,
                )
            result = self.run_command(
                f"test -f {shlex.quote(source)} || exit 17; "
                f"test ! -e {shlex.quote(target)} || exit 18; "
                f"mv -- {shlex.quote(source)} {shlex.quote(target)}",
                timeout=30,
            )
            if result.exit_code == 17:
                raise FileEditError(
                    f"Source file does not exist: {source}",
                    error_code="file_not_found",
                )
            if result.exit_code == 18:
                raise FileEditError(
                    f"Move destination already exists: {target}",
                    error_code="destination_exists",
                )
            self._require_success(result, f"Could not move {source} to {target}")
            changed_paths = self._verify_file_edit(source, target)
            return (
                f"Moved {source} to {target}. Changed paths: "
                f"{', '.join(changed_paths)}. Review the repository changes."
            )

        raise FileEditError(
            f"Unsupported edit operation: {operation}",
            error_code="invalid_operation",
            retryable=False,
        )

    def replace_text(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_replacements: int = 1,
        max_file_bytes: int = 2_000_000,
    ) -> str:
        """Apply an exact, count-checked UTF-8 replacement without diff parsing."""

        if not isinstance(old_text, str) or not old_text:
            raise TextReplaceError(
                "old_text must be a non-empty string",
                error_code="invalid_old_text",
                retryable=False,
            )
        if not isinstance(new_text, str):
            raise TextReplaceError(
                "new_text must be a string",
                error_code="invalid_new_text",
                retryable=False,
            )
        if old_text == new_text:
            raise TextReplaceError(
                "old_text and new_text are identical; no repository change was made",
                error_code="no_change",
                retryable=False,
            )
        if (
            not isinstance(expected_replacements, int)
            or isinstance(expected_replacements, bool)
            or expected_replacements < 1
            or max_file_bytes < 1
        ):
            raise TextReplaceError(
                "Replacement limits must be positive",
                error_code="invalid_replace_limits",
                retryable=False,
            )

        relative = self._resolve_path(path)
        absolute = posixpath.join(self.workdir.rstrip("/"), relative)
        try:
            stream, _ = self.container.get_archive(absolute)
            archive_bytes = b"".join(stream)
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as archive:
                members = archive.getmembers()
                if len(members) != 1 or not members[0].isfile():
                    raise TextReplaceError(
                        f"Path is not a regular file: {relative}",
                        error_code="not_a_text_file",
                        retryable=False,
                    )
                member = members[0]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise TextReplaceError(
                        f"Could not read file: {relative}",
                        error_code="file_read_failed",
                        retryable=False,
                    )
                payload = extracted.read(max_file_bytes + 1)
        except TextReplaceError:
            raise
        except Exception as error:
            raise TextReplaceError(
                f"Could not read file for replacement: {relative}: {error}",
                error_code="file_read_failed",
                retryable=False,
            ) from error

        if len(payload) > max_file_bytes:
            raise TextReplaceError(
                f"File exceeds the {max_file_bytes}-byte replacement limit: {relative}",
                error_code="file_too_large",
                retryable=False,
            )
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise TextReplaceError(
                f"File is not valid UTF-8 text: {relative}",
                error_code="not_a_text_file",
                retryable=False,
            ) from error

        actual_replacements = content.count(old_text)
        if actual_replacements != expected_replacements:
            error_code = (
                "text_not_found"
                if actual_replacements == 0
                else "replacement_count_mismatch"
            )
            diagnostic = ""
            if actual_replacements == 0:
                diagnostic = self._nearest_text_diagnostic(content, old_text)
            raise TextReplaceError(
                f"Expected {expected_replacements} exact occurrence(s) of old_text in "
                f"{relative}, found {actual_replacements}. Read the current raw file "
                "contents and retry with a unique exact block; the file was not changed."
                + (f"\n\n{diagnostic}" if diagnostic else ""),
                error_code=error_code,
            )

        updated = content.replace(old_text, new_text)
        updated_payload = updated.encode("utf-8")
        if len(updated_payload) > max_file_bytes:
            raise TextReplaceError(
                f"Updated file would exceed the {max_file_bytes}-byte replacement "
                f"limit: {relative}; the file was not changed.",
                error_code="replacement_too_large",
                retryable=False,
            )
        self._write_text_atomically(relative, updated_payload, source_member=member)
        return (
            f"Replaced {actual_replacements} exact occurrence(s) in {relative}. "
            "Review the resulting repository changes."
        )

    def _write_text_atomically(
        self,
        relative: str,
        payload: bytes,
        *,
        source_member: Optional[tarfile.TarInfo] = None,
        refuse_existing: bool = False,
    ) -> None:
        """Upload to a sibling temporary file and rename it into place."""

        absolute = posixpath.join(self.workdir.rstrip("/"), relative)
        parent = posixpath.dirname(absolute) or self.workdir
        temporary_name = f".agent-edit-{uuid.uuid4().hex}.tmp"
        temporary_relative = posixpath.join(
            posixpath.dirname(relative), temporary_name
        )
        output_archive = io.BytesIO()
        with tarfile.open(fileobj=output_archive, mode="w") as archive:
            output_member = tarfile.TarInfo(name=temporary_name)
            output_member.size = len(payload)
            output_member.mode = source_member.mode if source_member else 0o644
            if source_member is not None:
                output_member.uid = source_member.uid
                output_member.gid = source_member.gid
                output_member.mtime = source_member.mtime
            archive.addfile(output_member, io.BytesIO(payload))
        try:
            applied = self.container.put_archive(parent, output_archive.getvalue())
        except Exception as error:
            raise FileEditError(
                f"Could not stage edit for {relative}: {error}",
                error_code="file_write_failed",
                retryable=False,
            ) from error
        if applied is False:
            raise FileEditError(
                f"Could not stage edit for {relative}",
                error_code="file_write_failed",
                retryable=False,
            )

        target_check = (
            f"test ! -e {shlex.quote(relative)} || exit 18"
            if refuse_existing
            else f"test -f {shlex.quote(relative)} || exit 17"
        )
        result = self.run_command(
            f"{target_check}; mv -f -- {shlex.quote(temporary_relative)} "
            f"{shlex.quote(relative)}",
            timeout=30,
        )
        if result.exit_code != 0 or result.timed_out:
            self.run_command(
                f"rm -f -- {shlex.quote(temporary_relative)}",
                timeout=30,
            )
        if result.exit_code == 17:
            raise FileEditError(
                f"File does not exist or is not a regular file: {relative}",
                error_code="file_not_found",
            )
        if result.exit_code == 18:
            raise FileEditError(
                f"File already exists: {relative}",
                error_code="destination_exists",
            )
        self._require_success(result, f"Could not write {relative}")

    def _verify_file_edit(self, *paths: str) -> list[str]:
        quoted_paths = " ".join(shlex.quote(path) for path in paths)
        status = self.run_command(
            f"git status --short -- {quoted_paths}",
            timeout=30,
        )
        self._require_success(status, "Could not verify file edit")
        changed_paths = self._changed_paths_from_status(status.stdout)
        if not changed_paths:
            raise FileEditError(
                "The edit completed but produced no repository diff. Inspect the "
                "current file and choose an operation that changes it.",
                error_code="no_change",
            )
        return changed_paths

    @staticmethod
    def _nearest_text_diagnostic(content: str, old_text: str) -> str:
        """Return bounded current-file evidence without performing a fuzzy edit."""

        expected = old_text.splitlines()[:20]
        current = content.splitlines()
        if not expected or not current:
            return ""
        window_size = min(len(expected), len(current))
        best_index = 0
        best_ratio = -1.0
        for index in range(len(current) - window_size + 1):
            candidate = current[index : index + window_size]
            ratio = difflib.SequenceMatcher(
                None, expected[:window_size], candidate
            ).ratio()
            if ratio > best_ratio:
                best_index = index
                best_ratio = ratio
        excerpt = "\n".join(current[best_index : best_index + window_size])
        if len(excerpt) > 2_000:
            excerpt = excerpt[:2_000] + "\n[excerpt truncated]"
        return (
            f"Closest current block starts near line {best_index + 1}; copy exact "
            f"text from the file before retrying:\n{excerpt}"
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

    def get_patch(
        self,
        *,
        timeout: int = 30,
        max_output_chars: int = 1_000_000,
    ) -> str:
        intent = self.run_command("git add --intent-to-add -- .", timeout=timeout)
        self._require_success(intent, "Could not collect model patch")
        diff = self.run_command(
            f"git diff --binary {shlex.quote(self.base_commit)}",
            timeout=timeout,
            max_output_chars=max_output_chars + 1,
        )
        self._require_success(diff, "Could not collect model patch")
        check = self.run_command("git diff --check", timeout=timeout)
        self._require_success(check, "Model patch failed git diff --check")
        if len(diff.stdout) > max_output_chars:
            raise RepositoryRuntimeError(
                f"Model patch exceeds the {max_output_chars}-character limit"
            )
        return diff.stdout

    def _resolve_path(self, path: str) -> str:
        self.ensure_open()
        if not isinstance(path, str) or not path:
            raise RepositoryRuntimeError("Repository path must be a non-empty string")
        if path.startswith("/"):
            workdir = self.workdir.rstrip("/") or "/"
            if path == workdir or path == f"{workdir}/":
                path = "."
            elif workdir != "/" and path.startswith(f"{workdir}/"):
                path = path[len(workdir) + 1 :]
            else:
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

    @staticmethod
    def _changed_paths_from_status(status: str) -> list[str]:
        """Extract the paths Git reports as changed after a file edit."""

        changed_paths: list[str] = []
        for line in status.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip()
            if " -> " in path:
                path = path.rsplit(" -> ", 1)[1]
            if path:
                changed_paths.append(path)
        return sorted(set(changed_paths))

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
