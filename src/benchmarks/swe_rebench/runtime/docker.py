"""Docker container lifecycle for SWE-rebench inference runtimes."""

import re
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

        Docker SDK's synchronous ``exec_run`` does not enforce the supplied
        timeout. Hard timeout enforcement is added with the Docker operation
        backend in step 20; the value is validated here to keep the interface
        stable.
        """

        self.ensure_open()
        if not isinstance(command, str) or not command.strip():
            raise RepositoryRuntimeError("Command must be a non-empty string")
        if timeout < 1 or max_output_chars < 1:
            raise RepositoryRuntimeError("Command limits must be positive")

        started = time.monotonic()
        try:
            result = self.container.exec_run(
                ["/bin/sh", "-lc", command],
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
        return CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=self._decode(stdout)[:max_output_chars],
            stderr=self._decode(stderr)[:max_output_chars],
            duration_seconds=time.monotonic() - started,
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

    def list_files(self, path=".", **kwargs) -> str:
        raise RepositoryRuntimeError("Docker file operations require step 20")

    def read_file(self, path, **kwargs) -> str:
        raise RepositoryRuntimeError("Docker file operations require step 20")

    def search_code(self, query, **kwargs) -> str:
        raise RepositoryRuntimeError("Docker search operations require step 20")

    def apply_patch(self, patch, **kwargs) -> str:
        raise RepositoryRuntimeError("Docker patch operations require step 20")

    def get_diff(self, path=".", **kwargs) -> str:
        raise RepositoryRuntimeError("Docker diff operations require step 20")

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
