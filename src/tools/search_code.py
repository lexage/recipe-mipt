"""Ripgrep-backed search tool scoped to the active repository."""

from typing import Any, Dict, Optional

from src.benchmarks.swe_rebench.runtime import get_repository_runtime

from .base_tool import BaseTool


class SearchCodeTool(BaseTool):
    def __init__(
        self,
        name: str = "search_code",
        description: str = "Search text in files in the task repository.",
        max_results: int = 100,
        timeout: int = 30,
        max_output_chars: int = 30_000,
    ):
        super().__init__(name=name, description=description)
        if min(max_results, timeout, max_output_chars) < 1:
            raise ValueError("search limits must be positive")
        self.max_results = max_results
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
                        "query": {
                            "type": "string",
                            "description": "Literal text to find.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Repository-relative search root.",
                        },
                        "glob": {
                            "type": "string",
                            "description": "Optional file glob, for example *.py.",
                        },
                        "regex": {
                            "type": "boolean",
                            "description": "Interpret query as a regular expression.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def __call__(
        self,
        query: str,
        path: str = ".",
        glob: Optional[str] = None,
        regex: bool = False,
    ) -> str:
        return get_repository_runtime().search_code(
            query,
            path=path,
            glob=glob,
            regex=regex,
            max_results=self.max_results,
            timeout=self.timeout,
            max_output_chars=self.max_output_chars,
        )
