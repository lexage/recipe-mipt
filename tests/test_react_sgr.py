import logging
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.tools import ToolResult

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

    def __init__(self, result="README.md"):
        self.calls = []
        self.result = result

    def get_prompt_description(self):
        return "list_files: list repository files"

    def __call__(self, **arguments):
        self.calls.append(arguments)
        return self.result


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


def finish_step():
    step = AgentStep(
        thought="Patch and validation are complete",
        action="finish",
        action_input={},
        is_final=True,
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

    def test_length_failure_gets_concise_recovery_instruction(self):
        client, _ = fake_openai_client()
        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="model",
                tools=[FakeTool()],
            )
        agent._last_llm_error = "LengthFinishReasonError: output limit"
        agent._last_llm_error_code = "response_too_long"

        agent._handle_invalid_response(0)

        feedback = agent.memory[-1]["content"]
        self.assertIn("exceeded the model output limit", feedback)
        self.assertIn("one short JSON step", feedback)

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
        self.assertNotIn("max_tokens", completions.parse.call_args.kwargs)
        self.assertEqual(tool.calls, [{}])

    def test_max_tokens_is_forwarded_to_step_and_final_answer_requests(self):
        client, completions = fake_openai_client()
        completions.parse.return_value = list_files_step()
        completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done"))]
        )

        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="Qwen/Qwen2.5-32B-Instruct",
                tools=[FakeTool()],
                max_iterations=1,
                max_tokens=1024,
            )

        with self.assertLogs(level=logging.INFO):
            agent.run("Fix the repository")
        self.assertEqual(completions.parse.call_args.kwargs["max_tokens"], 1024)

        agent._generate_final_answer("Fix the repository")
        self.assertEqual(completions.create.call_args.kwargs["max_tokens"], 1024)

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
        self.assertNotIn("finish immediately", REACT_SYSTEM_PROMPT)
        self.assertIn("validation support the solution", REACT_SYSTEM_PROMPT)

    def test_system_prompt_contains_only_configured_tool_descriptions(self):
        client, _ = fake_openai_client()

        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="model",
                tools=[FakeTool()],
            )

        self.assertIn("list_files: list repository files", agent.instruction)
        self.assertIn("Action MUST be one of: list_files or 'finish'", agent.instruction)
        self.assertNotIn("replace_text", agent.instruction)
        self.assertNotIn("apply_patch", agent.instruction)

    def test_tool_results_preserve_generic_failure_status(self):
        client, _ = fake_openai_client()
        failure = ToolResult.error(
            "operation failed",
            error_code="transient_failure",
        )
        tool = FakeTool(result=failure)

        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="model",
                tools=[tool],
            )

        result = agent.execute_tool("list_files", {})
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "transient_failure")
        self.assertEqual(str(result), "operation failed")

    def test_repeated_generic_tool_failures_block_only_the_same_call(self):
        client, _ = fake_openai_client()
        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="model",
                tools=[FakeTool()],
            )

        step = AgentStep(
            thought="try",
            action="list_files",
            action_input={"path": "src"},
        )
        failure = ToolResult.error("failed", error_code="same_failure")
        agent._record_failed_execution(step, failure)
        signature = agent._call_signature(step.action, step.action_input)
        self.assertNotIn(signature, agent._blocked_call_signatures)
        agent._record_failed_execution(step, failure)
        self.assertIn(signature, agent._blocked_call_signatures)

        different_step = AgentStep(
            thought="inspect elsewhere",
            action="list_files",
            action_input={"path": "tests"},
        )
        self.assertNotIn(
            agent._call_signature(
                different_step.action, different_step.action_input
            ),
            agent._blocked_call_signatures,
        )

    def test_non_retryable_failure_blocks_identical_call_immediately(self):
        client, _ = fake_openai_client()
        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="model",
                tools=[FakeTool()],
            )
        step = AgentStep(thought="try", action="list_files", action_input={})
        agent._record_failed_execution(
            step,
            ToolResult.error(
                "not supported",
                error_code="unsupported",
                retryable=False,
            ),
        )
        self.assertIn(
            agent._call_signature(step.action, step.action_input),
            agent._blocked_call_signatures,
        )

    def test_history_context_counts_complete_react_turns(self):
        client, _ = fake_openai_client()
        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="model",
                tools=[FakeTool()],
                history_context=2,
            )
        agent.memory = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        for index in range(3):
            agent.memory.extend(
                [
                    {"role": "assistant", "content": f"action-{index}"},
                    {"role": "user", "content": f"observation-{index}"},
                ]
            )
        messages = agent._get_sliding_window_messages()
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[2]["content"], "action-1")
        self.assertEqual(messages[-1]["content"], "observation-2")

    def test_patch_hunk_line_number_changes_are_same_strategy(self):
        client, _ = fake_openai_client()
        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="model",
                tools=[FakeTool()],
            )
        first = {"patch": "--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-old\n+new"}
        second = {"patch": "--- a/a.py\n+++ b/a.py\n@@ -20,1 +20,1 @@\n-old\n+new"}
        self.assertEqual(
            agent._call_signature("apply_patch", first),
            agent._call_signature("apply_patch", second),
        )

    def test_finish_can_skip_redundant_synthesis_request(self):
        client, completions = fake_openai_client()
        completions.parse.return_value = finish_step()
        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="model",
                tools=[FakeTool()],
                synthesize_final_answer=False,
            )
        with self.assertLogs(level=logging.INFO):
            result = agent.run("Fix the repository")
        self.assertEqual(result, "Patch and validation are complete")
        completions.create.assert_not_called()

    def test_finalization_fields_are_derived_from_action(self):
        tool_step = AgentStep(
            thought="inspect",
            action="list_files",
            action_input={},
            is_final=True,
        )
        finish = AgentStep(
            thought="done",
            action="finish",
            action_input={"stale": "value"},
            is_final=False,
        )
        self.assertFalse(tool_step.is_final)
        self.assertTrue(finish.is_final)
        self.assertEqual(finish.action_input, {})

    def test_tool_json_schema_is_enforced_when_available(self):
        from src.tools import ReadFileTool

        client, _ = fake_openai_client()
        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="model",
                tools=[ReadFileTool()],
            )

        valid, error = agent._validate_tool_args(
            "read_file", {"path": "/testbed/src/a.py", "unknown": 1}
        )
        self.assertFalse(valid)
        self.assertIn("Unexpected argument", error)
        valid, error = agent._validate_tool_args(
            "read_file", {"path": "/testbed/src/a.py"}
        )
        self.assertFalse(valid)
        self.assertIn("repository-relative", error)

    def test_finish_guard_requires_current_revision_evidence(self):
        client, _ = fake_openai_client()
        with patch("src.agents.pipelines.react_sgr.OpenAI", return_value=client):
            agent = ReActAgentSGR(
                url="http://localhost:11455/v1",
                model_name="model",
                tools=[FakeTool()],
                require_repository_change_before_finish=True,
                require_diff_before_finish=True,
                require_validation_before_finish=True,
            )

        self.assertIn("repository change", agent._finish_guard_error())
        agent._repository_revision = 1
        agent._last_diff_revision = 1
        agent._last_validation_revision = 1
        self.assertIsNone(agent._finish_guard_error())
        agent._repository_revision = 2
        self.assertIn("current full diff", agent._finish_guard_error())

    def test_invalid_few_shot_type_and_limits_are_rejected(self):
        parameters = (
            {"max_iterations": 0},
            {"history_context": 0},
            {"max_tokens": 0},
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
