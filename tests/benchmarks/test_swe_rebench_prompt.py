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
        self.assertIn("complete definition", prompt)
        self.assertIn("exact target symbol", prompt)
        self.assertIn("concrete causal hypothesis", prompt)
        self.assertIn("missing, empty, optional, and default inputs", prompt)
        self.assertIn("stop repeating it and change strategy", prompt)
        self.assertIn("narrowest dependency-light test or check", prompt)
        self.assertIn("minimal local reproduction", prompt)
        self.assertIn("Review the final git diff", prompt)
        self.assertIn("patch is non-empty", prompt)
        self.assertIn("Attempt relevant validation when available", prompt)
        self.assertNotIn("validation evidence supports it", prompt)
        self.assertIn("do not install arbitrary", prompt)
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
