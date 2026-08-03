"""Tool for validating and applying unified diffs to the active repository."""

import shlex
import subprocess
from typing import Any, Dict

from .base_tool import BaseTool
from .repository_context import (
    RepositoryPathError,
    get_repository_context,
    resolve_repository_path,
)


class ApplyPatchTool(BaseTool):
    def __init__(
        self,
        name: str = "apply_patch",
        description: str = "Validate and apply a unified git diff to the task repository.",
        timeout: int = 30,
        max_patch_bytes: int = 1_000_000,
    ):
        super().__init__(name=name, description=description)
        if timeout < 1 or max_patch_bytes < 1:
            raise ValueError("patch limits must be positive")
        self.timeout = timeout
        self.max_patch_bytes = max_patch_bytes

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch": {
                            "type": "string",
                            "description": "Unified diff to apply.",
                        }
                    },
                    "required": ["patch"],
                },
            },
        }

    def _validate_paths(self, patch: str) -> list[str]:
        paths: list[str] = []
        for line in patch.splitlines():
            if not line.startswith("diff --git "):
                continue
            try:
                parts = shlex.split(line)
            except ValueError as error:
                raise RepositoryPathError("Malformed diff header") from error
            if len(parts) != 4:
                raise RepositoryPathError("Malformed diff header")
            for raw_path, prefix in ((parts[2], "a/"), (parts[3], "b/")):
                if not raw_path.startswith(prefix):
                    raise RepositoryPathError(f"Unexpected diff path: {raw_path}")
                relative = raw_path[len(prefix) :]
                resolve_repository_path(relative)
                paths.append(relative)
        if not paths:
            raise RepositoryPathError("Patch has no 'diff --git' file headers")
        return sorted(set(paths))

    def __call__(self, patch: str) -> str:
        if not patch.strip():
            return "Patch must not be empty"
        encoded = patch.encode("utf-8")
        if len(encoded) > self.max_patch_bytes:
            return f"Patch exceeds the {self.max_patch_bytes}-byte limit"
        try:
            paths = self._validate_paths(patch)
        except RepositoryPathError as error:
            return f"Patch rejected: {error}"

        context = get_repository_context()
        try:
            check = subprocess.run(
                ["git", "apply", "--check", "-"],
                cwd=context.root,
                input=patch,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            if check.returncode != 0:
                detail = check.stderr.strip() or check.stdout.strip()
                return f"Patch check failed: {detail}"
            applied = subprocess.run(
                ["git", "apply", "-"],
                cwd=context.root,
                input=patch,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError:
            return "Patch failed: git is not installed"
        except subprocess.TimeoutExpired:
            return f"Patch command timed out after {self.timeout} seconds"

        if applied.returncode != 0:
            detail = applied.stderr.strip() or applied.stdout.strip()
            return f"Patch apply failed: {detail}"
        return "Patch applied successfully.\n\nChanged paths:\n" + "\n".join(
            f"- {path}" for path in paths
        )
