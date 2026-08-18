import logging
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.agents.pipelines.react_sgr import AgentStep, ReActAgentSGR


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

    def test_request_failure_is_logged_and_returned_after_retry_limit(self):
        client, completions = fake_openai_client()
        completions.parse.side_effect = RuntimeError("404 Not Found")

        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="Qwen/Qwen2.5-32B-Instruct",
                tools=[FakeTool()],
                max_iterations=30,
                max_retries=2,
            )

        with self.assertLogs(level=logging.INFO) as captured:
            result = agent.run("Fix the repository")

        log_text = "\n".join(captured.output)
        self.assertEqual(completions.parse.call_count, 2)
        self.assertIn("LLM_REQUEST", log_text)
        self.assertIn("LLM_REQUEST_FAILED", log_text)
        self.assertIn("LLM_RETRY", log_text)
        self.assertIn("failed_attempts=2", log_text)
        self.assertIn("RuntimeError: 404 Not Found", result)

    def test_successful_structured_response_is_logged_before_tool_execution(self):
        client, completions = fake_openai_client()
        tool = FakeTool()
        step = AgentStep(
            thought="Inspect the repository",
            action="list_files",
            action_input={},
        )
        completions.parse.return_value = SimpleNamespace(
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

        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="Qwen/Qwen2.5-32B-Instruct",
                tools=[tool],
                max_iterations=1,
            )

        with self.assertLogs(level=logging.INFO) as captured:
            agent.run("Fix the repository")

        log_text = "\n".join(captured.output)
        self.assertIn("LLM_RESPONSE", log_text)
        self.assertIn('"action":"list_files"', log_text)
        self.assertEqual(tool.calls, [{}])

    def test_retry_and_iteration_limits_must_be_positive(self):
        for parameter in ({"max_iterations": 0}, {"max_retries": 0}):
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
