"""Tool for running a bounded non-interactive command in the task repository."""

import os
import shlex
import subprocess
import time
from typing import Any, Dict

from .base_tool import BaseTool
from .repository_context import get_repository_context


class RunCommandTool(BaseTool):
    def __init__(
        self,
        name: str = "run_command",
        description: str = "Run a non-interactive command in the task repository.",
        default_timeout: int = 120,
        max_timeout: int = 600,
        max_output_chars: int = 30_000,
    ):
        super().__init__(name=name, description=description)
        if min(default_timeout, max_timeout, max_output_chars) < 1:
            raise ValueError("command limits must be positive")
        if default_timeout > max_timeout:
            raise ValueError("default_timeout must not exceed max_timeout")
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.max_output_chars = max_output_chars

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command and arguments.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds.",
                        },
                    },
                    "required": ["command"],
                },
            },
        }

    @staticmethod
    def _environment(repository_root: str) -> dict[str, str]:
        allowed = ("PATH", "LANG", "LC_ALL", "TERM", "PYTHONPATH")
        environment = {key: os.environ[key] for key in allowed if key in os.environ}
        environment.update({"HOME": repository_root, "GIT_TERMINAL_PROMPT": "0"})
        return environment

    def __call__(self, command: str, timeout: int | None = None) -> str:
        try:
            arguments = shlex.split(command)
        except ValueError as error:
            return f"Invalid command: {error}"
        if not arguments:
            return "Command must not be empty"

        effective_timeout = self.default_timeout if timeout is None else timeout
        if effective_timeout < 1 or effective_timeout > self.max_timeout:
            return f"Timeout must be between 1 and {self.max_timeout} seconds"

        context = get_repository_context()
        started = time.monotonic()
        try:
            result = subprocess.run(
                arguments,
                cwd=context.root,
                env=self._environment(str(context.root)),
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                timeout=effective_timeout,
                check=False,
            )
        except FileNotFoundError:
            return f"Command not found: {arguments[0]}"
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            return self._render(
                command, None, time.monotonic() - started, stdout, stderr, True
            )

        return self._render(
            command,
            result.returncode,
            time.monotonic() - started,
            result.stdout,
            result.stderr,
            False,
        )

    def _render(
        self,
        command: str,
        exit_code: int | None,
        duration: float,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> str:
        output = (
            f"Command: {command}\n"
            f"Exit code: {'timeout' if timed_out else exit_code}\n"
            f"Duration: {duration:.3f}s\n\n"
            f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        )
        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + "\n[output truncated]"
        return output
