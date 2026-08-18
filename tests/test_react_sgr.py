import logging
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.agents.pipelines.react_sgr import (
    AgentConfig,
    AgentStep,
    FINISH_PROMPT_TEMPLATE,
    REACT_SYSTEM_PROMPT,
    ReActAgentSGR,
)


class FakeTool:
    name = "list_files"
    arg_schema = None

    def __init__(self):
        self.calls = []

    def get_prompt_description(self):
        return "list_files: list repository files"

    def __call__(self, **arguments):
        self.calls.append(arguments)
        return "README.md"


def fake_openai_client():
    completions = Mock()
    return (
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        completions,
    )


def list_files_step():
    step = AgentStep(
        thought="Inspect the repository",
        action="list_files",
        action_input={},
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=step,
                    content=step.model_dump_json(),
                ),
                finish_reason="stop",
            )
        ]
    )


class ReActAgentSGRDiagnosticsTests(unittest.TestCase):
    def test_models_resource_url_is_normalized_to_api_base(self):
        client, _ = fake_openai_client()

        with patch(
            "src.agents.pipelines.react_sgr.OpenAI", return_value=client
        ) as openai:
            with self.assertLogs(level=logging.WARNING) as captured:
                agent = ReActAgentSGR(
                    url="http://localhost:11455/v1/models/",
                    model_name="Qwen/Qwen2.5-32B-Instruct",
                    tools=[FakeTool()],
                )

        self.assertEqual(agent.base_url, "http://localhost:11455/v1")
        openai.assert_called_once_with(
            base_url="http://localhost:11455/v1", api_key="vllm"
        )
        self.assertIn("LLM_BASE_URL_NORMALIZED", "\n".join(captured.output))

    def test_request_failure_is_logged_without_changing_retry_input(self):
        client, completions = fake_openai_client()
        completions.parse.side_effect = RuntimeError("404 Not Found")

        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="Qwen/Qwen2.5-32B-Instruct",
                tools=[FakeTool()],
                max_iterations=30,
            )

        with patch.object(AgentConfig, "MAX_RETRY_COUNT", 2):
            with self.assertLogs(level=logging.INFO) as captured:
                result = agent.run("Fix the repository")

        log_text = "\n".join(captured.output)
        self.assertEqual(completions.parse.call_count, 2)
        self.assertIn("LLM_REQUEST", log_text)
        self.assertIn("LLM_REQUEST_FAILED", log_text)
        self.assertIn("LLM_RETRY", log_text)
        self.assertIn("failed_attempts=2", log_text)
        self.assertIn("RuntimeError: 404 Not Found", log_text)
        self.assertNotIn("404 Not Found", result)
        self.assertIn(
            "Failed to parse or validate JSON response",
            agent.memory[-1]["content"],
        )
        self.assertNotIn("404 Not Found", agent.memory[-1]["content"])

    def test_zero_shot_preserves_baseline_first_user_message(self):
        client, completions = fake_openai_client()
        tool = FakeTool()
        completions.parse.return_value = list_files_step()

        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="Qwen/Qwen2.5-32B-Instruct",
                tools=[tool],
                max_iterations=1,
                few_shot_type="zero_shot",
            )

        with self.assertLogs(level=logging.INFO):
            agent.run("Fix the repository")

        messages = completions.parse.call_args.kwargs["messages"]
        self.assertEqual(messages[1]["content"], "Fix the repository")
        self.assertEqual(tool.calls, [{}])

    def test_cot_examples_are_prepended_to_the_task(self):
        client, completions = fake_openai_client()
        completions.parse.return_value = list_files_step()

        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="Qwen/Qwen2.5-32B-Instruct",
                tools=[FakeTool()],
                max_iterations=1,
                few_shot_type="cot",
            )

        with self.assertLogs(level=logging.INFO):
            agent.run("Fix the repository")

        messages = completions.parse.call_args.kwargs["messages"]
        self.assertTrue(messages[1]["content"].endswith("Fix the repository"))
        self.assertNotEqual(messages[1]["content"], "Fix the repository")

    def test_baseline_prompts_are_preserved(self):
        self.assertIn(
            "Few-shot examples may demonstrate different tools",
            REACT_SYSTEM_PROMPT,
        )
        self.assertLess(
            FINISH_PROMPT_TEMPLATE.index("CONVERSATION HISTORY:"),
            FINISH_PROMPT_TEMPLATE.index("TASK:"),
        )

    def test_invalid_few_shot_type_and_limits_are_rejected(self):
        parameters = (
            {"max_iterations": 0},
            {"history_context": 0},
            {"few_shot_type": "missing"},
        )
        for parameter in parameters:
            with self.subTest(parameter=parameter):
                with self.assertRaises(ValueError):
                    ReActAgentSGR(
                        url="http://localhost:11455/v1",
                        model_name="model",
                        tools=[FakeTool()],
                        **parameter,
                    )


if __name__ == "__main__":
    unittest.main()
