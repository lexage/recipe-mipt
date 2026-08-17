import sys
import types
import unittest

# src.tools exposes LLMTool, whose optional client is irrelevant to these tests.
openai_stub = types.ModuleType("openai")
openai_stub.OpenAI = object
sys.modules.setdefault("openai", openai_stub)

from src.benchmarks.swe_rebench.runtime import (  # noqa: E402
    CommandResult,
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
)


class RecordingRuntime(RepositoryRuntime):
    def __init__(self):
        super().__init__("owner__repo-1", "deadbeef", "/testbed")
        self.calls = []
        self.command_result = CommandResult("pytest", 0, "done", "", 0.25)

    def list_files(self, path=".", **kwargs):
        self.calls.append(("list_files", path, kwargs))
        return "listed"

    def read_file(self, path, **kwargs):
        self.calls.append(("read_file", path, kwargs))
        return "read"

    def search_code(self, query, **kwargs):
        self.calls.append(("search_code", query, kwargs))
        return "found"

    def apply_patch(self, patch, **kwargs):
        self.calls.append(("apply_patch", patch, kwargs))
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
            self.assertEqual(ApplyPatchTool()("diff"), "applied")
            self.assertIn("Exit code: 0", RunCommandTool()("pytest"))
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
        self.assertIn("between 1 and 2", invalid)

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
