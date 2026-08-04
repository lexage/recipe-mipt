"""Tool for reading bounded line ranges from repository files."""

from typing import Any, Dict

from .base_tool import BaseTool
from .repository_context import resolve_repository_path


class ReadFileTool(BaseTool):
    def __init__(
        self,
        name: str = "read_file",
        description: str = "Read a line range from a text file in the task repository.",
        max_lines: int = 400,
        max_file_bytes: int = 2_000_000,
    ):
        super().__init__(name=name, description=description)
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
                    },
                    "required": ["path"],
                },
            },
        }

    def __call__(self, path: str, start_line: int = 1, end_line: int = 200) -> str:
        if start_line < 1 or end_line < start_line:
            return "Invalid line range: require 1 <= start_line <= end_line"
        if end_line - start_line + 1 > self.max_lines:
            end_line = start_line + self.max_lines - 1

        file_path = resolve_repository_path(path)
        if not file_path.is_file():
            return f"File does not exist: {path}"
        if file_path.stat().st_size > self.max_file_bytes:
            return f"File is too large to read: {path}"

        data = file_path.read_bytes()
        if b"\x00" in data:
            return f"Binary file cannot be read: {path}"
        try:
            lines = data.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            return f"File is not valid UTF-8 text: {path}"

        selected = lines[start_line - 1 : end_line]
        rendered = "\n".join(
            f"{number:>6} | {line}"
            for number, line in enumerate(selected, start=start_line)
        )
        actual_end = start_line + len(selected) - 1
        if not selected:
            actual_end = start_line - 1
            rendered = "[no lines in requested range]"
        return (
            f"File: {path}\nLines: {start_line}-{actual_end} of {len(lines)}\n\n"
            f"{rendered}"
        )
