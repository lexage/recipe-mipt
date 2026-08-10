import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.benchmarks.swe_rebench.runtime import (
    LocalRepositoryRuntime,
    RepositoryRuntimeClosedError,
    RepositoryRuntimeError,
)


class LocalRepositoryRuntimeTests(unittest.TestCase):
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
            ["git", "config", "user.name", "Runtime Tests"],
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
        self.runtime = LocalRepositoryRuntime(self.root, "owner__repo-1", base_commit)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_file_listing_reading_and_search(self) -> None:
        listing = self.runtime.list_files("src")
        content = self.runtime.read_file("src/calculator.py", start_line=2, end_line=2)
        matches = self.runtime.search_code("left - right", glob="*.py")

        self.assertIn("calculator.py", listing)
        self.assertIn("2 |     return left - right", content)
        self.assertIn("src/calculator.py:2", matches)

    def test_patch_command_and_diff_work_end_to_end(self) -> None:
        patch = """diff --git a/src/calculator.py b/src/calculator.py
--- a/src/calculator.py
+++ b/src/calculator.py
@@ -1,2 +1,2 @@
 def add(left, right):
-    return left - right
+    return left + right
"""
        applied = self.runtime.apply_patch(patch)
        command = self.runtime.run_command(
            f'{sys.executable} -c "from src.calculator import add; assert add(2, 3) == 5"'
        )
        diff = self.runtime.get_diff()
        model_patch = self.runtime.get_patch()

        self.assertIn("Patch applied successfully", applied)
        self.assertEqual(command.exit_code, 0)
        self.assertFalse(command.timed_out)
        self.assertIn("return left + right", diff)
        self.assertTrue(model_patch.startswith("diff --git"))
        self.assertNotIn("Status:", model_patch)

    def test_diff_includes_new_files(self) -> None:
        (self.root / "new.py").write_text("created = True\n")
        diff = self.runtime.get_diff()
        self.assertIn("new.py", diff)
        self.assertIn("created = True", diff)

    def test_command_timeout_is_normalized(self) -> None:
        result = self.runtime.run_command(
            f"{sys.executable} -c \"import time; print('started', flush=True); time.sleep(2)\"",
            timeout=1,
        )
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)
        self.assertIn("started", result.stdout)

    def test_paths_cannot_escape_checkout(self) -> None:
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        (self.root / "external").symlink_to(outside, target_is_directory=True)
        for path in ("../outside", ".git/config", "external/file.py"):
            with self.subTest(path=path):
                with self.assertRaises(RepositoryRuntimeError):
                    self.runtime.resolve_path(path)

    def test_closed_runtime_rejects_operations(self) -> None:
        self.runtime.close()
        with self.assertRaises(RepositoryRuntimeClosedError):
            self.runtime.list_files()

    def test_constructor_rejects_missing_root(self) -> None:
        with self.assertRaises(RepositoryRuntimeError):
            LocalRepositoryRuntime(self.root / "missing", "owner__repo-2", "deadbeef")


if __name__ == "__main__":
    unittest.main()
