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
            "Use a/ and b/ path prefixes. Inside every hunk, each line must begin "
            "with exactly one prefix character: space for unchanged context, '-' "
            "for removed text, or '+' for added text. Copy context exactly from the "
            "current raw file, including indentation and punctuation; never include "
            "display line-number prefixes. Prefer a small hunk with two to four exact "
            "surrounding lines. Hunk counts are recounted automatically, so changing "
            "only line numbers cannot repair stale or malformed content. After a "
            "failure, inspect the reported current-file evidence and construct a new "
            "diff from it. Success means the patch changed the repository, not that "
            "the behavior is correct."
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
                                "+++ b/path, and an @@ hunk header. Context lines must "
                                "match the current file exactly. Do not use Markdown "
                                "code fences."
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
