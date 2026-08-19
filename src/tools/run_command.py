"""Tool for running a bounded non-interactive command in the task repository."""

import re

from typing import Any, Dict

from src.benchmarks.swe_rebench.runtime import CommandResult, get_repository_runtime

from .base_tool import BaseTool, ToolResult


class RunCommandTool(BaseTool):
    _MUTATING_COMMAND_PATTERNS = (
        re.compile(r"(?:^|[;&|]\s*)(?:mv|cp|rm|touch|truncate)\s"),
        re.compile(r"\b(?:sed|perl)\b[^\n;&|]*\s-i(?:\s|$)"),
        re.compile(
            r"\bgit\s+(?:add|checkout|clean|commit|merge|rebase|reset|restore|switch)\b"
        ),
    )
    _VALIDATION_COMMAND_PATTERNS = (
        re.compile(r"(?:^|\s)(?:pytest|tox|nox)(?:\s|$)"),
        re.compile(r"\bpython(?:\d+(?:\.\d+)?)?\s+-m\s+(?:pytest|unittest|compileall)\b"),
        re.compile(r"(?:^|\s)(?:npm|pnpm|yarn)\s+(?:run\s+)?test(?:\s|$)"),
        re.compile(r"(?:^|\s)(?:cargo|go)\s+test(?:\s|$)"),
        re.compile(r"(?:^|\s)(?:make|cmake\s+--build\s+\S+\s+--)\s*(?:test|check)(?:\s|$)"),
        re.compile(r"\bgit\s+diff\s+--check\b"),
    )
    _PACKAGE_INSTALL_PATTERNS = (
        re.compile(
            r"\b(?:python(?:\d+(?:\.\d+)?)?\s+-m\s+)?"
            r"pip(?:\d+(?:\.\d+)?)?\s+install\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:conda|mamba|micromamba)\s+install\b", re.IGNORECASE),
        re.compile(
            r"\b(?:apt|apt-get|apk|dnf|yum)\s+(?:add|install)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:npm|pnpm|yarn)\s+(?:add|ci|install)\b", re.IGNORECASE),
        re.compile(r"\b(?:poetry|uv)\s+add\b|\buv\s+pip\s+install\b", re.IGNORECASE),
        re.compile(
            r"\b(?:gem|cargo|go)\s+install\b|\bcargo\s+fetch\b|"
            r"\bgo\s+mod\s+download\b",
            re.IGNORECASE,
        ),
    )

    def __init__(
        self,
        name: str = "run_command",
        description: str = (
            "Run a non-interactive inspection, build, or test command in the task "
            "repository. Use an editing tool for edits; do not create branches or "
            "commits."
        ),
        default_timeout: int = 120,
        max_timeout: int = 600,
        max_output_chars: int = 30_000,
        allow_package_install: bool = True,
    ):
        super().__init__(name=name, description=description)
        if min(default_timeout, max_timeout, max_output_chars) < 1:
            raise ValueError("command limits must be positive")
        if default_timeout > max_timeout:
            raise ValueError("default_timeout must not exceed max_timeout")
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.max_output_chars = max_output_chars
        self.allow_package_install = allow_package_install

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

    def __call__(self, command: str, timeout: int | None = None) -> ToolResult:
        if not self.allow_package_install and any(
            pattern.search(command) for pattern in self._PACKAGE_INSTALL_PATTERNS
        ):
            return ToolResult.error(
                "Package installation is disabled for this experiment. Use the "
                "dependencies already present in the task image and run a focused "
                "test or minimal local reproduction instead.",
                error_code="package_install_command",
                retryable=False,
            )
        if any(pattern.search(command) for pattern in self._MUTATING_COMMAND_PATTERNS):
            return ToolResult.error(
                "Repository-editing shell commands are not allowed. Use apply_patch "
                "for file changes so edits can be validated and tracked.",
                error_code="repository_edit_command",
                retryable=False,
            )
        is_validation = any(
            pattern.search(command) for pattern in self._VALIDATION_COMMAND_PATTERNS
        )
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
            return ToolResult.error(
                output,
                error_code="command_timeout",
                validation_status="failed" if is_validation else None,
            )
        if result.exit_code != 0:
            error_code = self._nonzero_error_code(command, result.exit_code)
            return ToolResult.error(
                output,
                error_code=error_code,
                validation_status="failed" if is_validation else None,
            )
        return ToolResult.ok(
            output,
            validation_status="passed" if is_validation else None,
        )

    @staticmethod
    def _nonzero_error_code(command: str, exit_code: int | None) -> str:
        if re.search(r"(?:^|\s)(?:pytest|python(?:\d+(?:\.\d+)?)?\s+-m\s+pytest)(?:\s|$)", command):
            return {
                1: "tests_failed",
                2: "pytest_interrupted",
                3: "pytest_internal_error",
                4: "pytest_usage_error",
                5: "no_tests_collected",
            }.get(exit_code, "validation_failed")
        return "nonzero_exit_code"

    @staticmethod
    def _render(result: CommandResult) -> str:
        return (
            f"Exit code: {'timeout' if result.timed_out else result.exit_code}\n"
            f"Duration: {result.duration_seconds:.3f}s\n\n"
            f"Command: {result.command}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
