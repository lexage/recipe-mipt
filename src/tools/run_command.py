"""Tool for running a bounded non-interactive command in the task repository."""

from typing import Any, Dict

from src.benchmarks.swe_rebench.runtime import CommandResult, get_repository_runtime

from .base_tool import BaseTool


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

    def __call__(self, command: str, timeout: int | None = None) -> str:
        effective_timeout = self.default_timeout if timeout is None else timeout
        if effective_timeout < 1 or effective_timeout > self.max_timeout:
            return f"Timeout must be between 1 and {self.max_timeout} seconds"
        result = get_repository_runtime().run_command(
            command,
            timeout=effective_timeout,
            max_output_chars=self.max_output_chars,
        )
        output = self._render(result)
        if len(output) > self.max_output_chars:
            marker = "\n[output truncated]"
            return output[: self.max_output_chars - len(marker)] + marker
        return output

    @staticmethod
    def _render(result: CommandResult) -> str:
        return (
            f"Exit code: {'timeout' if result.timed_out else result.exit_code}\n"
            f"Duration: {result.duration_seconds:.3f}s\n\n"
            f"Command: {result.command}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
