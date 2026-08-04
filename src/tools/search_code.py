"""Ripgrep-backed search tool scoped to the active repository."""

import subprocess
from typing import Any, Dict, Optional

from .base_tool import BaseTool
from .repository_context import get_repository_context, resolve_repository_path


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
        if not query:
            return "Search query must not be empty"
        search_root = resolve_repository_path(path)
        if not search_root.exists():
            return f"Search path does not exist: {path}"

        context = get_repository_context()
        relative_root = search_root.relative_to(context.root)
        pathspec = "." if not relative_root.parts else relative_root.as_posix()
        command = ["rg", "--line-number", "--column", "--color", "never"]
        if not regex:
            command.append("--fixed-strings")
        if glob:
            command.extend(["--glob", glob])
        command.extend(["--glob", "!.git/**", "--", query, pathspec])

        try:
            result = subprocess.run(
                command,
                cwd=context.root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError:
            return "Search failed: ripgrep (rg) is not installed"
        except subprocess.TimeoutExpired:
            return f"Search timed out after {self.timeout} seconds"

        if result.returncode == 1:
            return "No matches found"
        if result.returncode != 0:
            return f"Search failed (exit {result.returncode}): {result.stderr.strip()}"

        lines = result.stdout.splitlines()
        truncated = len(lines) > self.max_results
        output = "\n".join(lines[: self.max_results])
        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars]
            truncated = True
        if truncated:
            output += "\n[output truncated]"
        return output
