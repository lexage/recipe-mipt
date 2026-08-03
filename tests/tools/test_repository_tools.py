import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

# src.tools exposes LLMTool, whose optional client is irrelevant to these tests.
openai_stub = types.ModuleType("openai")
openai_stub.OpenAI = object
sys.modules.setdefault("openai", openai_stub)

from src.tools import (  # noqa: E402
    ApplyPatchTool,
    GitDiffTool,
    ListFilesTool,
    ReadFileTool,
    RepositoryContext,
    RunCommandTool,
    SearchCodeTool,
    bind_repository_context,
)


class RepositoryToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "repository"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "tests@example.com"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Repository Tools Tests"],
            cwd=self.root,
            check=True,
        )
        source = self.root / "src" / "calculator.py"
        source.parent.mkdir()
        source.write_text("def add(left, right):\n    return left - right\n")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "initial"], cwd=self.root, check=True)
        base_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        self.context = RepositoryContext(self.root, "owner__repo-1", base_commit)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_list_files_is_sorted_and_bounded(self) -> None:
        (self.root / ".secret").write_text("hidden")
        with bind_repository_context(self.context):
            output = ListFilesTool(max_entries=10)(".")
        self.assertIn("src/", output)
        self.assertIn("src/calculator.py", output)
        self.assertNotIn(".git", output)
        self.assertNotIn(".secret", output)

    def test_read_file_returns_numbered_range_and_rejects_binary(self) -> None:
        (self.root / "binary.dat").write_bytes(b"a\x00b")
        with bind_repository_context(self.context):
            output = ReadFileTool()("src/calculator.py", start_line=2, end_line=2)
            binary = ReadFileTool()("binary.dat")
        self.assertIn("2 |     return left - right", output)
        self.assertIn("Binary file", binary)

    def test_search_code_supports_literal_queries_and_globs(self) -> None:
        with bind_repository_context(self.context):
            output = SearchCodeTool()("left - right", glob="*.py")
            missing = SearchCodeTool()("not present")
        self.assertIn("src/calculator.py:2", output)
        self.assertEqual(missing, "No matches found")

    def test_apply_patch_checks_and_applies_diff(self) -> None:
        patch = """diff --git a/src/calculator.py b/src/calculator.py
--- a/src/calculator.py
+++ b/src/calculator.py
@@ -1,2 +1,2 @@
 def add(left, right):
-    return left - right
+    return left + right
"""
        with bind_repository_context(self.context):
            result = ApplyPatchTool()(patch)
        self.assertIn("Patch applied successfully", result)
        self.assertIn(
            "return left + right", (self.root / "src/calculator.py").read_text()
        )

    def test_apply_patch_rejects_paths_outside_repository(self) -> None:
        patch = """diff --git a/../outside.py b/../outside.py
--- a/../outside.py
+++ b/../outside.py
@@ -0,0 +1 @@
+bad = True
"""
        with bind_repository_context(self.context):
            result = ApplyPatchTool()(patch)
        self.assertIn("Patch rejected", result)

    def test_run_command_uses_repository_as_working_directory(self) -> None:
        command = f'{sys.executable} -c "import pathlib; print(pathlib.Path.cwd())"'
        with bind_repository_context(self.context):
            output = RunCommandTool()(command)
        self.assertIn("Exit code: 0", output)
        self.assertIn(str(self.root), output)

    def test_git_diff_reports_changes_from_base_commit(self) -> None:
        (self.root / "src" / "calculator.py").write_text(
            "def add(left, right):\n    return left + right\n"
        )
        with bind_repository_context(self.context):
            output = GitDiffTool()()
        self.assertIn("M src/calculator.py", output)
        self.assertIn("return left + right", output)


if __name__ == "__main__":
    unittest.main()
