"""Tool for listing files in the active task repository."""

from typing import Any, Dict

from src.benchmarks.swe_rebench.runtime import get_repository_runtime

from .base_tool import BaseTool


class ListFilesTool(BaseTool):
    def __init__(
        self,
        name: str = "list_files",
        description: str = "List files and directories in the task repository.",
        max_depth: int = 3,
        max_entries: int = 200,
        include_hidden: bool = False,
    ):
        super().__init__(
            name=name,
            description=description,
            repository_path_args=("path",),
        )
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
        return get_repository_runtime().list_files(
            path,
            max_depth=self.max_depth,
            max_entries=self.max_entries,
            include_hidden=self.include_hidden,
        )
