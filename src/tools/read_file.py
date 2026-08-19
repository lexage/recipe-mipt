"""Tool for reading bounded line ranges from repository files."""

from typing import Any, Dict

from src.benchmarks.swe_rebench.runtime import get_repository_runtime

from .base_tool import BaseTool


class ReadFileTool(BaseTool):
    def __init__(
        self,
        name: str = "read_file",
        description: str = (
            "Read a raw line range from a text file in the task repository. "
            "Set line_numbers=true only when display line numbers help navigation."
        ),
        max_lines: int = 400,
        max_file_bytes: int = 2_000_000,
    ):
        super().__init__(
            name=name,
            description=description,
            repository_path_args=("path",),
        )
        if max_lines < 1 or max_file_bytes < 1:
            raise ValueError("read limits must be positive")
        self.max_lines = max_lines
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
                            "description": "Repository-relative file.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "First line, starting at 1.",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "Last line, inclusive.",
                        },
                        "line_numbers": {
                            "type": "boolean",
                            "description": (
                                "Include display-only line-number prefixes. Defaults "
                                "to false so returned file text can be copied exactly."
                            ),
                        },
                    },
                    "required": ["path"],
                },
            },
        }

    def __call__(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 200,
        line_numbers: bool = False,
    ) -> str:
        return get_repository_runtime().read_file(
            path,
            start_line=start_line,
            end_line=end_line,
            max_lines=self.max_lines,
            max_file_bytes=self.max_file_bytes,
            line_numbers=line_numbers,
        )
