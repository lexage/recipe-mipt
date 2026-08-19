import sys
import types
import unittest

# src.tools exposes LLMTool, whose optional client is irrelevant to these tests.
openai_stub = types.ModuleType("openai")
openai_stub.OpenAI = object
sys.modules.setdefault("openai", openai_stub)

from src.benchmarks.swe_rebench.runtime import (  # noqa: E402
    CommandResult,
    PatchApplyError,
    RepositoryRuntime,
    bind_repository_runtime,
)
from src.tools import (  # noqa: E402
    ApplyPatchTool,
    GitDiffTool,
    ListFilesTool,
    ReadFileTool,
    RunCommandTool,
    SearchCodeTool,
    ToolResult,
)


class RecordingRuntime(RepositoryRuntime):
    def __init__(self):
        super().__init__("owner__repo-1", "deadbeef", "/testbed")
        self.calls = []
        self.command_result = CommandResult("pytest", 0, "done", "", 0.25)
        self.patch_error = None
        self.search_result = "found"

    def list_files(self, path=".", **kwargs):
        self.calls.append(("list_files", path, kwargs))
        return "listed"

    def read_file(self, path, **kwargs):
        self.calls.append(("read_file", path, kwargs))
        return "read"

    def search_code(self, query, **kwargs):
        self.calls.append(("search_code", query, kwargs))
        return self.search_result

    def apply_patch(self, patch, **kwargs):
        self.calls.append(("apply_patch", patch, kwargs))
        if self.patch_error is not None:
            raise self.patch_error
        return "applied"

    def run_command(self, command, **kwargs):
        self.calls.append(("run_command", command, kwargs))
        return CommandResult(
            command,
            self.command_result.exit_code,
            self.command_result.stdout,
            self.command_result.stderr,
            self.command_result.duration_seconds,
            self.command_result.timed_out,
        )

    def get_diff(self, path=".", **kwargs):
        self.calls.append(("get_diff", path, kwargs))
        return "diffed"

    def get_patch(self, **kwargs):
        return "patch"


class RepositoryToolDelegationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = RecordingRuntime()

    def test_all_tools_delegate_to_bound_runtime(self) -> None:
        with bind_repository_runtime(self.runtime):
            self.assertEqual(ListFilesTool()("src"), "listed")
            self.assertEqual(ReadFileTool()("src/a.py", 2, 3), "read")
            self.assertEqual(SearchCodeTool()("symbol", glob="*.py"), "found")
            patch_result = ApplyPatchTool()("diff")
            command_result = RunCommandTool()("pytest")
            self.assertEqual(patch_result, "applied")
            self.assertTrue(patch_result.success)
            self.assertTrue(patch_result.progress)
            self.assertIn("Exit code: 0", command_result)
            self.assertTrue(command_result.success)
            self.assertEqual(GitDiffTool()(), "diffed")

        self.assertEqual(
            [call[0] for call in self.runtime.calls],
            [
                "list_files",
                "read_file",
                "search_code",
                "apply_patch",
                "run_command",
                "get_diff",
            ],
        )
        self.assertEqual(self.runtime.calls[1][2]["start_line"], 2)
        self.assertEqual(self.runtime.calls[1][2]["end_line"], 3)
        self.assertEqual(self.runtime.calls[2][2]["glob"], "*.py")
        self.assertEqual(self.runtime.calls[4][2]["timeout"], 120)

    def test_run_command_formats_timeout_and_rejects_excessive_timeout(self) -> None:
        self.runtime.command_result = CommandResult(
            "sleep 10", None, "started", "", 1.0, timed_out=True
        )
        tool = RunCommandTool(default_timeout=1, max_timeout=2)

        with bind_repository_runtime(self.runtime):
            timeout = tool("sleep 10")
            invalid = tool("sleep 10", timeout=3)

        self.assertIn("Exit code: timeout", timeout)
        self.assertIn("started", timeout)
        self.assertFalse(timeout.success)
        self.assertEqual(timeout.error_code, "command_timeout")
        self.assertIn("between 1 and 2", invalid)
        self.assertFalse(invalid.success)
        self.assertEqual(invalid.error_code, "invalid_timeout")

    def test_apply_patch_schema_contains_valid_unified_diff_shape(self) -> None:
        tool = ApplyPatchTool()
        description = tool.get_schema()["function"]["description"]
        patch_description = tool.get_schema()["function"]["parameters"]["properties"][
            "patch"
        ]["description"]

        self.assertIn("--- a/path/to/file.py", description)
        self.assertIn("+++ b/path/to/file.py", description)
        self.assertIn("@@ -1,1 +1,1 @@", description)
        self.assertIn("--- a/path", patch_description)
        self.assertIn("+++ b/path", patch_description)
        self.assertIn("@@ hunk header", patch_description)

    def test_patch_and_nonzero_command_report_failures(self) -> None:
        self.runtime.patch_error = PatchApplyError(
            "Patch check failed: corrupt patch",
            error_code="malformed_diff",
        )
        self.runtime.command_result = CommandResult(
            "pytest", 2, "", "collection failed", 0.1
        )

        with bind_repository_runtime(self.runtime):
            patch_result = ApplyPatchTool()("broken diff")
            command_result = RunCommandTool()("pytest")

        self.assertIsInstance(patch_result, ToolResult)
        self.assertFalse(patch_result.success)
        self.assertEqual(patch_result.error_code, "malformed_diff")
        self.assertFalse(command_result.success)
        self.assertEqual(command_result.error_code, "nonzero_exit_code")

    def test_empty_search_result_is_success_without_progress(self) -> None:
        self.runtime.search_result = "No matches found"
        with bind_repository_runtime(self.runtime):
            result = SearchCodeTool()("missing symbol")
        self.assertTrue(result.success)
        self.assertFalse(result.progress)

    def test_search_timeout_is_retryable_failure(self) -> None:
        self.runtime.search_result = "Search timed out after 30 seconds"
        with bind_repository_runtime(self.runtime):
            result = SearchCodeTool()("symbol")
        self.assertFalse(result.success)
        self.assertTrue(result.retryable)
        self.assertEqual(result.error_code, "search_timeout")

    def test_run_command_truncates_large_output(self) -> None:
        self.runtime.command_result = CommandResult(
            "command", 0, "x" * 500, "failure", 0.1
        )
        with bind_repository_runtime(self.runtime):
            output = RunCommandTool(max_output_chars=120)("command")

        self.assertLessEqual(len(output), 120)
        self.assertIn("output truncated", output)

    def test_tool_configuration_limits_are_validated(self) -> None:
        constructors = (
            lambda: ListFilesTool(max_entries=0),
            lambda: ReadFileTool(max_lines=0),
            lambda: SearchCodeTool(timeout=0),
            lambda: ApplyPatchTool(max_patch_bytes=0),
            lambda: RunCommandTool(default_timeout=3, max_timeout=2),
            lambda: GitDiffTool(timeout=0),
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()


if __name__ == "__main__":
    unittest.main()
