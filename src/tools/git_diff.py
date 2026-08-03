"""Tool for inspecting changes in the active task repository."""

import subprocess
from typing import Any, Dict

from .base_tool import BaseTool
from .repository_context import get_repository_context, resolve_repository_path


class GitDiffTool(BaseTool):
    def __init__(
        self,
        name: str = "git_diff",
        description: str = "Show repository status and the diff from the task base commit.",
        timeout: int = 30,
        max_output_chars: int = 50_000,
    ):
        super().__init__(name=name, description=description)
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
        context = get_repository_context()
        selected = resolve_repository_path(path)
        relative = selected.relative_to(context.root)
        pathspec = "." if not relative.parts else relative.as_posix()

        try:
            status = subprocess.run(
                ["git", "status", "--short", "--", pathspec],
                cwd=context.root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            command = ["git", "diff", "--binary"]
            if stat_only:
                command.append("--stat")
            command.extend([context.base_commit, "--", pathspec])
            diff = subprocess.run(
                command,
                cwd=context.root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError:
            return "Diff failed: git is not installed"
        except subprocess.TimeoutExpired:
            return f"Diff command timed out after {self.timeout} seconds"

        if status.returncode != 0 or diff.returncode != 0:
            detail = status.stderr.strip() or diff.stderr.strip()
            return f"Diff failed: {detail}"
        status_text = status.stdout.strip() or "[clean]"
        diff_text = diff.stdout.strip() or "[no diff]"
        output = f"Status:\n{status_text}\n\nDiff:\n{diff_text}"
        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + "\n[output truncated]"
        return output
