"""Tool for inspecting changes in the active task repository."""

from typing import Any, Dict

from src.benchmarks.swe_rebench.runtime import get_repository_runtime

from .base_tool import BaseTool


class GitDiffTool(BaseTool):
    def __init__(
        self,
        name: str = "git_diff",
        description: str = "Show repository status and the diff from the task base commit.",
        timeout: int = 30,
        max_output_chars: int = 50_000,
    ):
        super().__init__(
            name=name,
            description=description,
            provides_diff=True,
            repository_path_args=("path",),
        )
        if timeout < 1 or max_output_chars < 1:
            raise ValueError("diff limits must be positive")
        self.timeout = timeout
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
                        "path": {
                            "type": "string",
                            "description": "Optional repository-relative path.",
                        },
                        "stat_only": {
                            "type": "boolean",
                            "description": "Return only the diff summary.",
                        },
                    },
                },
            },
        }

    def __call__(self, path: str = ".", stat_only: bool = False) -> str:
        return get_repository_runtime().get_diff(
            path,
            stat_only=stat_only,
            timeout=self.timeout,
            max_output_chars=self.max_output_chars,
        )
