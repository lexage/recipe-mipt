"""Tool for validating and applying unified diffs to the active repository."""

from typing import Any, Dict

from src.benchmarks.swe_rebench.runtime import (
    PatchApplyError,
    RepositoryRuntimeError,
    get_repository_runtime,
)

from .base_tool import BaseTool, ToolResult


class ApplyPatchTool(BaseTool):
    def __init__(
        self,
        name: str = "apply_patch",
        description: str = (
            "Validate and apply a raw unified diff to the task repository. "
            "Do not wrap the diff in Markdown fences. A minimal valid patch is:\n"
            "--- a/path/to/file.py\n"
            "+++ b/path/to/file.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old line\n"
            "+new line\n"
            "For one localized exact replacement in an existing file, prefer "
            "replace_text when available because it does not require diff syntax. Copy unchanged "
            "context lines exactly from the latest raw read_file output. "
            "After a failed patch, read the target file again and change the strategy "
            "instead of only changing hunk line numbers. Use this tool, not shell "
            "redirection or sed -i, to edit files."
        ),
        timeout: int = 30,
        max_patch_bytes: int = 1_000_000,
    ):
        super().__init__(
            name=name,
            description=description,
            mutates_repository=True,
        )
        if timeout < 1 or max_patch_bytes < 1:
            raise ValueError("patch limits must be positive")
        self.timeout = timeout
        self.max_patch_bytes = max_patch_bytes

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch": {
                            "type": "string",
                            "description": (
                                "Raw unified diff to apply. Include --- a/path, "
                                "+++ b/path, and an @@ hunk header. Do not use "
                                "Markdown code fences."
                            ),
                        }
                    },
                    "required": ["patch"],
                },
            },
        }

    def __call__(self, patch: str) -> ToolResult:
        try:
            result = get_repository_runtime().apply_patch(
                patch,
                timeout=self.timeout,
                max_patch_bytes=self.max_patch_bytes,
            )
        except PatchApplyError as error:
            return ToolResult.error(
                str(error),
                error_code=error.error_code,
                retryable=error.retryable,
            )
        except RepositoryRuntimeError as error:
            return ToolResult.error(
                str(error),
                error_code="repository_runtime_error",
                retryable=False,
            )
        return ToolResult.ok(result, progress=True)
