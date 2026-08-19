"""Tool for running a bounded non-interactive command in the task repository."""

from typing import Any, Dict

from src.benchmarks.swe_rebench.runtime import CommandResult, get_repository_runtime

from .base_tool import BaseTool, ToolResult


class RunCommandTool(BaseTool):
    def __init__(
        self,
        name: str = "run_command",
        description: str = (
            "Run a non-interactive inspection, build, or test command in the task "
            "repository. Use apply_patch for edits; do not create branches or commits."
        ),
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
                            "description": (
                                "Inspection, build, or test command and arguments. "
                                "Do not edit files or run git checkout/commit."
                            ),
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

    def __call__(
        self, command: str, timeout: int | None = None
    ) -> ToolResult:
        effective_timeout = self.default_timeout if timeout is None else timeout
        if effective_timeout < 1 or effective_timeout > self.max_timeout:
            return ToolResult.error(
                f"Timeout must be between 1 and {self.max_timeout} seconds",
                error_code="invalid_timeout",
            )
        result = get_repository_runtime().run_command(
            command,
            timeout=effective_timeout,
            max_output_chars=self.max_output_chars,
        )
        output = self._render(result)
        if len(output) > self.max_output_chars:
            marker = "\n[output truncated]"
            output = output[: self.max_output_chars - len(marker)] + marker
        if result.timed_out:
            return ToolResult.error(output, error_code="command_timeout")
        if result.exit_code != 0:
            return ToolResult.error(output, error_code="nonzero_exit_code")
        return ToolResult.ok(output)

    @staticmethod
    def _render(result: CommandResult) -> str:
        return (
            f"Exit code: {'timeout' if result.timed_out else result.exit_code}\n"
            f"Duration: {result.duration_seconds:.3f}s\n\n"
            f"Command: {result.command}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
