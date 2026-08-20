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
        self.assertIn("Before editing, establish a baseline", prompt)
        self.assertIn("existing focused test", prompt)
        self.assertIn("minimal reproduction", prompt)
        self.assertIn("do not create replacement test files", prompt)
        self.assertIn("complete target implementation", prompt)
        self.assertIn("smallest source change", prompt)
        self.assertIn("rerun the same baseline command", prompt)
        self.assertIn("zero collected tests", prompt)
        self.assertIn("does not validate the change", prompt)
        self.assertIn("relevant regression tests", prompt)
        self.assertIn("Review the final git diff", prompt)
        self.assertIn("not that the behavior is correct", prompt)
        self.assertIn("validation is genuinely unavailable", prompt)
        self.assertIn("patch is non-empty", prompt)
        self.assertIn("do not install arbitrary", prompt)
        self.assertIn("Do not add or modify tests", prompt)
        self.assertIn("Do not create commits or branches", prompt)
        self.assertNotIn("Look at parser.py", prompt)

    def test_hints_are_opt_in(self):
        prompt = SWERebenchPromptBuilder(include_hints=True).build(self.task)
        self.assertIn("Hints:\nLook at parser.py", prompt)

    def test_invalid_workdir_is_rejected(self):
        with self.assertRaises(ValueError):
            SWERebenchPromptBuilder().build(self.task, "testbed")


if __name__ == "__main__":
    unittest.main()
