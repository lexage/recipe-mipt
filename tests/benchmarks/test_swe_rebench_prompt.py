import unittest

from src.benchmarks.swe_rebench import SWERebenchPromptBuilder, SWERebenchTask


class SWERebenchPromptBuilderTests(unittest.TestCase):
    def setUp(self):
        self.task = SWERebenchTask(
            "owner__repo-1",
            "owner/repo",
            "deadbeef",
            "Fix the parser.",
            hints_text="Look at parser.py",
        )

    def test_prompt_describes_container_task_without_reference_data(self):
        prompt = SWERebenchPromptBuilder().build(self.task)
        self.assertIn("owner__repo-1", prompt)
        self.assertIn("owner/repo", prompt)
        self.assertIn("deadbeef", prompt)
        self.assertIn("/testbed", prompt)
        self.assertIn("smallest repository patch", prompt)
        self.assertIn("evidence or reproduction examples", prompt)
        self.assertIn("source code, existing tests", prompt)
        self.assertIn("call sites before editing", prompt)
        self.assertIn("complete relevant functions and nearby tests", prompt)
        self.assertIn("Temporary reproduction code is allowed", prompt)
        self.assertIn("remove it before finishing", prompt)
        self.assertIn("do not turn production modules into debug scripts", prompt)
        self.assertIn("smallest source-code change", prompt)
        self.assertIn("collected no tests", prompt)
        self.assertIn("exercised no relevant behavior", prompt)
        self.assertIn("not successful validation", prompt)
        self.assertIn("Review the final git diff", prompt)
        self.assertIn("remove unintended changes", prompt)
        self.assertIn("do not install arbitrary", prompt)
        self.assertIn("Do not add or modify tests unless", prompt)
        self.assertIn("Temporary reproduction artifacts must not remain", prompt)
        self.assertIn("Do not create commits or branches", prompt)
        self.assertNotIn("Before editing, establish a baseline", prompt)
        self.assertNotIn("rerun the same baseline command", prompt)
        self.assertNotIn("Finish only after", prompt)
        self.assertNotIn("patch is non-empty", prompt)
        self.assertNotIn("Look at parser.py", prompt)

    def test_hints_are_opt_in(self):
        prompt = SWERebenchPromptBuilder(include_hints=True).build(self.task)
        self.assertIn("Hints:\nLook at parser.py", prompt)

    def test_invalid_workdir_is_rejected(self):
        with self.assertRaises(ValueError):
            SWERebenchPromptBuilder().build(self.task, "testbed")


if __name__ == "__main__":
    unittest.main()
