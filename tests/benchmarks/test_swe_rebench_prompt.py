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
        self.assertIn("Apply the smallest correct fix", prompt)
        self.assertNotIn("Look at parser.py", prompt)

    def test_hints_are_opt_in(self):
        prompt = SWERebenchPromptBuilder(include_hints=True).build(self.task)
        self.assertIn("Hints:\nLook at parser.py", prompt)

    def test_invalid_workdir_is_rejected(self):
        with self.assertRaises(ValueError):
            SWERebenchPromptBuilder().build(self.task, "testbed")


if __name__ == "__main__":
    unittest.main()
