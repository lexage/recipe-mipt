"""Structured repository editing tool exposed under the stable apply_patch name."""

from typing import Any, Dict, Optional

from src.benchmarks.swe_rebench.runtime import (
    FileEditError,
    RepositoryRuntimeError,
    get_repository_runtime,
)

from .base_tool import BaseTool, ToolResult


class ApplyPatchTool(BaseTool):
    """Apply one exact file operation without asking the model to encode a diff."""

    _OPERATION_ARGUMENTS = {
        "replace": {"operation", "path", "old_text", "new_text"},
        "create": {"operation", "path", "new_text"},
        "delete": {"operation", "path"},
        "move": {"operation", "path", "destination"},
    }

    def __init__(
        self,
        name: str = "apply_patch",
        description: str = (
            "Apply one structured file edit to the task repository. Choose exactly "
            "one operation: replace an exact unique block in an existing UTF-8 file; "
            "create a new UTF-8 file; delete an existing file; or move an existing "
            "file. For replace, copy old_text exactly from the current file, including "
            "indentation and punctuation, and provide the complete replacement as "
            "new_text. Use one operation per call. This tool does not accept unified "
            "diff syntax or Markdown fences. After editing, inspect the repository "
            "changes and run relevant tests with the configured tools."
        ),
        max_file_bytes: int = 2_000_000,
    ):
        super().__init__(
            name=name,
            description=description,
            repository_path_args=("path", "destination"),
            mutates_repository=True,
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
                        "operation": {
                            "type": "string",
                            "enum": ["replace", "create", "delete", "move"],
                            "description": (
                                "Single file operation to perform. Must be one of: "
                                "replace, create, delete, move."
                            ),
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "Repository-relative source or target path. Absolute "
                                "paths and paths containing '..' are rejected."
                            ),
                        },
                        "old_text": {
                            "type": "string",
                            "description": (
                                "For replace only: a non-empty exact block copied "
                                "from the current file. It must occur exactly once."
                            ),
                        },
                        "new_text": {
                            "type": "string",
                            "description": (
                                "For replace: the complete replacement block. For "
                                "create: the complete UTF-8 file contents."
                            ),
                        },
                        "destination": {
                            "type": "string",
                            "description": (
                                "For move only: a repository-relative destination "
                                "path that does not already exist."
                            ),
                        },
                    },
                    "required": ["operation", "path"],
                    "additionalProperties": False,
                },
            },
        }

    def validate_arguments(self, arguments: Dict[str, Any]) -> Optional[str]:
        error = super().validate_arguments(arguments)
        if error:
            return error

        operation = arguments["operation"]
        allowed = self._OPERATION_ARGUMENTS.get(operation)
        if allowed is None:
            choices = ", ".join(self._OPERATION_ARGUMENTS)
            return f"Argument 'operation' must be one of: {choices}"
        unexpected = sorted(set(arguments) - allowed)
        if unexpected:
            return (
                f"Operation '{operation}' does not accept argument(s): "
                + ", ".join(unexpected)
            )

        if operation == "replace":
            if not arguments.get("old_text"):
                return "Operation 'replace' requires non-empty old_text"
            if "new_text" not in arguments:
                return "Operation 'replace' requires new_text"
            if arguments["old_text"] == arguments["new_text"]:
                return "old_text and new_text must differ"
        elif operation == "create" and "new_text" not in arguments:
            return "Operation 'create' requires new_text"
        elif operation == "move":
            if not arguments.get("destination"):
                return "Operation 'move' requires destination"
            if arguments["destination"] == arguments["path"]:
                return "Move destination must differ from path"
        return None

    def __call__(
        self,
        operation: str,
        path: str,
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        destination: Optional[str] = None,
    ) -> ToolResult:
        arguments = {
            "operation": operation,
            "path": path,
            **({"old_text": old_text} if old_text is not None else {}),
            **({"new_text": new_text} if new_text is not None else {}),
            **({"destination": destination} if destination is not None else {}),
        }
        validation_error = self.validate_arguments(arguments)
        if validation_error:
            return ToolResult.error(
                validation_error,
                error_code="invalid_edit_arguments",
                retryable=True,
            )
        try:
            result = get_repository_runtime().apply_file_edit(
                operation,
                path,
                old_text=old_text,
                new_text=new_text,
                destination=destination,
                max_file_bytes=self.max_file_bytes,
            )
        except FileEditError as error:
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
