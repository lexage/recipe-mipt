import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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
    RepositoryPathError,
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

    def test_list_files_honors_depth_and_entry_limits(self) -> None:
        nested = self.root / "src" / "nested"
        nested.mkdir()
        (nested / "deep.py").write_text("pass\n")
        (self.root / "another.py").write_text("pass\n")
        with bind_repository_context(self.context):
            shallow = ListFilesTool(max_depth=1)("src")
            bounded = ListFilesTool(max_entries=1)(".")
            not_directory = ListFilesTool()("src/calculator.py")
        self.assertIn("nested/", shallow)
        self.assertNotIn("deep.py", shallow)
        self.assertIn("output truncated", bounded)
        self.assertIn("not a directory", not_directory)

    def test_read_file_returns_numbered_range_and_rejects_binary(self) -> None:
        (self.root / "binary.dat").write_bytes(b"a\x00b")
        with bind_repository_context(self.context):
            output = ReadFileTool()("src/calculator.py", start_line=2, end_line=2)
            binary = ReadFileTool()("binary.dat")
        self.assertIn("2 |     return left - right", output)
        self.assertIn("Binary file", binary)

    def test_read_file_enforces_ranges_size_and_utf8(self) -> None:
        (self.root / "large.txt").write_text("too large")
        (self.root / "invalid.txt").write_bytes(b"\xff\xfe")
        with bind_repository_context(self.context):
            invalid_range = ReadFileTool()("src/calculator.py", 3, 2)
            limited = ReadFileTool(max_lines=1)("src/calculator.py", 1, 20)
            large = ReadFileTool(max_file_bytes=2)("large.txt")
            invalid_utf8 = ReadFileTool()("invalid.txt")
        self.assertIn("Invalid line range", invalid_range)
        self.assertIn("Lines: 1-1", limited)
        self.assertIn("too large", large)
        self.assertIn("not valid UTF-8", invalid_utf8)

    def test_search_code_supports_literal_queries_and_globs(self) -> None:
        with bind_repository_context(self.context):
            output = SearchCodeTool()("left - right", glob="*.py")
            missing = SearchCodeTool()("not present")
        self.assertIn("src/calculator.py:2", output)
        self.assertEqual(missing, "No matches found")

    def test_search_code_supports_regex_and_truncates_results(self) -> None:
        (self.root / "src" / "second.py").write_text("left + right\nleft * right\n")
        with bind_repository_context(self.context):
            output = SearchCodeTool(max_results=1)(
                r"left . right", glob="*.py", regex=True
            )
        self.assertIn("src/", output)
        self.assertIn("output truncated", output)

    def test_search_code_reports_missing_binary_and_timeout(self) -> None:
        with bind_repository_context(self.context):
            with patch(
                "src.tools.search_code.subprocess.run", side_effect=FileNotFoundError
            ):
                missing = SearchCodeTool()("value")
            with patch(
                "src.tools.search_code.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["rg"], 1),
            ):
                timeout = SearchCodeTool(timeout=1)("value")
        self.assertIn("not installed", missing)
        self.assertIn("timed out", timeout)

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

    def test_apply_patch_creates_and_deletes_files(self) -> None:
        create_patch = """diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1 @@
+created = True
"""
        delete_patch = """diff --git a/src/calculator.py b/src/calculator.py
deleted file mode 100644
--- a/src/calculator.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def add(left, right):
-    return left - right
"""
        with bind_repository_context(self.context):
            created = ApplyPatchTool()(create_patch)
            deleted = ApplyPatchTool()(delete_patch)
        self.assertIn("successfully", created)
        self.assertEqual((self.root / "new.py").read_text(), "created = True\n")
        self.assertIn("successfully", deleted)
        self.assertFalse((self.root / "src" / "calculator.py").exists())

    def test_apply_patch_failure_does_not_modify_file(self) -> None:
        original = (self.root / "src" / "calculator.py").read_text()
        bad_patch = """diff --git a/src/calculator.py b/src/calculator.py
--- a/src/calculator.py
+++ b/src/calculator.py
@@ -1,2 +1,2 @@
-content that is not present
+replacement
"""
        with bind_repository_context(self.context):
            result = ApplyPatchTool()(bad_patch)
        self.assertIn("Patch check failed", result)
        self.assertEqual((self.root / "src" / "calculator.py").read_text(), original)

    def test_run_command_uses_repository_as_working_directory(self) -> None:
        command = f'{sys.executable} -c "import pathlib; print(pathlib.Path.cwd())"'
        with bind_repository_context(self.context):
            output = RunCommandTool()(command)
        self.assertIn("Exit code: 0", output)
        self.assertIn(str(self.root), output)

    def test_run_command_reports_failure_stderr_and_truncation(self) -> None:
        command = (
            f"{sys.executable} -c \"import sys; print('x' * 100); "
            "print('failure', file=sys.stderr); sys.exit(3)\""
        )
        with bind_repository_context(self.context):
            output = RunCommandTool(max_output_chars=120)(command)
        self.assertIn("Exit code: 3", output)
        self.assertIn("output truncated", output)

    def test_run_command_reports_timeout_and_invalid_timeout(self) -> None:
        command = f"{sys.executable} -c \"import time; print('started', flush=True); time.sleep(2)\""
        with bind_repository_context(self.context):
            timeout = RunCommandTool(default_timeout=1, max_timeout=2)(command)
            invalid = RunCommandTool(default_timeout=1, max_timeout=2)(
                command, timeout=3
            )
        self.assertIn("Exit code: timeout", timeout)
        self.assertIn("started", timeout)
        self.assertIn("between 1 and 2", invalid)

    def test_git_diff_reports_changes_from_base_commit(self) -> None:
        (self.root / "src" / "calculator.py").write_text(
            "def add(left, right):\n    return left + right\n"
        )
        with bind_repository_context(self.context):
            output = GitDiffTool()()
        self.assertIn("M src/calculator.py", output)
        self.assertIn("return left + right", output)

    def test_git_diff_reports_clean_added_deleted_and_stat(self) -> None:
        with bind_repository_context(self.context):
            clean = GitDiffTool()()
        self.assertIn("[clean]", clean)
        self.assertIn("[no diff]", clean)

        (self.root / "new.py").write_text("new = True\n")
        (self.root / "src" / "calculator.py").unlink()
        with bind_repository_context(self.context):
            output = GitDiffTool()()
            stat = GitDiffTool()(stat_only=True)
        self.assertIn("new.py", output)
        self.assertIn("deleted file", output)
        self.assertIn("files changed", stat)

    def test_git_diff_rejects_path_escape(self) -> None:
        with bind_repository_context(self.context):
            with self.assertRaises(RepositoryPathError):
                GitDiffTool()("../outside")


if __name__ == "__main__":
    unittest.main()
