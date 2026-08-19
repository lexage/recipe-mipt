"""Tool for safe exact-text replacements in repository files."""

from typing import Any, Dict

from src.benchmarks.swe_rebench.runtime import (
    RepositoryRuntimeError,
    TextReplaceError,
    get_repository_runtime,
)

from .base_tool import BaseTool, ToolResult


class ReplaceTextTool(BaseTool):
    """Replace a uniquely identified text block without unified-diff syntax."""

    def __init__(
        self,
        name: str = "replace_text",
        description: str = (
            "Replace exact text in one existing UTF-8 file. Use this for a localized "
            "edit when the old text can be copied exactly from current raw file "
            "contents. The file is changed only if the exact occurrence count "
            "matches. This tool does not create new files and is not intended for "
            "structural or multi-file edits."
        ),
        max_file_bytes: int = 2_000_000,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            mutates_repository=True,
            repository_path_args=("path",),
        )
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")
        self.max_file_bytes = max_file_bytes

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Repository-relative path to an existing UTF-8 file.",
                        },
                        "old_text": {
                            "type": "string",
                            "description": (
                                "Exact current file text to replace, copied without "
                                "display line numbers or read_file metadata."
                            ),
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text. An empty string deletes old_text.",
                        },
                        "expected_replacements": {
                            "type": "integer",
                            "description": (
                                "Required exact occurrence count. Defaults to 1; a "
                                "mismatch leaves the file unchanged."
                            ),
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        }

    def __call__(
        self,
        path: str,
        old_text: str,
        new_text: str,
        expected_replacements: int = 1,
    ) -> ToolResult:
        try:
            result = get_repository_runtime().replace_text(
                path,
                old_text,
                new_text,
                expected_replacements=expected_replacements,
                max_file_bytes=self.max_file_bytes,
            )
        except TextReplaceError as error:
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
