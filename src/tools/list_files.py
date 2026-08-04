"""Tool for listing files in the active task repository."""

from pathlib import Path
from typing import Any, Dict

from .base_tool import BaseTool
from .repository_context import resolve_repository_path


class ListFilesTool(BaseTool):
    def __init__(
        self,
        name: str = "list_files",
        description: str = "List files and directories in the task repository.",
        max_depth: int = 3,
        max_entries: int = 200,
        include_hidden: bool = False,
    ):
        super().__init__(name=name, description=description)
        if max_depth < 0 or max_entries < 1:
            raise ValueError("max_depth must be non-negative and max_entries positive")
        self.max_depth = max_depth
        self.max_entries = max_entries
        self.include_hidden = include_hidden

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
                            "description": "Repository-relative directory to list.",
                        }
                    },
                },
            },
        }

    def __call__(self, path: str = ".") -> str:
        directory = resolve_repository_path(path)
        if not directory.exists():
            return f"Path does not exist: {path}"
        if not directory.is_dir():
            return f"Path is not a directory: {path}"

        entries: list[str] = []
        truncated = False
        for candidate in sorted(directory.rglob("*")):
            relative = candidate.relative_to(directory)
            if len(relative.parts) > self.max_depth:
                continue
            if not self.include_hidden and any(
                part.startswith(".") for part in relative.parts
            ):
                continue
            suffix = "/" if candidate.is_dir() else ""
            entries.append(f"{relative.as_posix()}{suffix}")
            if len(entries) == self.max_entries:
                truncated = True
                break

        header = f"Directory: {Path(path).as_posix()}"
        body = "\n".join(entries) if entries else "[empty]"
        footer = f"\n\n[{len(entries)} entries"
        if truncated:
            footer += ", output truncated"
        return f"{header}\n\n{body}{footer}]"
