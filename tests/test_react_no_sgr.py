"""Offline regression tests; no model server, database, or optional RAG stack needed.

Run: python -m unittest discover -s tests -p test_react_no_sgr.py -v
Requires the repository's openai, pydantic, PyYAML and networkx dependencies.
"""

import importlib
import importlib.util
import json
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.pipelines.configs import ConfigLoader
from src.pipelines.pipeline_builder import PipelineBuilder
from src.pipelines.registry import ComponentRegistry


ROOT = Path(__file__).resolve().parents[1]


def load_agent_module():
    # Package __init__ files eagerly import unrelated RAG/ML dependencies.
    # Load the real agent, defaults and real BaseTool/LLMTool in isolation.
    with patch.dict(sys.modules):
        tools_package = types.ModuleType("src.tools")
        tools_package.__path__ = [str(ROOT / "src/tools")]
        sys.modules["src.tools"] = tools_package
        tools_package.BaseTool = importlib.import_module("src.tools.base_tool").BaseTool
        tools_package.LLMTool = importlib.import_module("src.tools.llm_tool").LLMTool
        package = types.ModuleType("_react_ablation_tests")
        package.__path__ = [str(ROOT / "src/agents/pipelines")]
        sys.modules[package.__name__] = package
        return importlib.import_module("_react_ablation_tests.react_no_sgr")


react = load_agent_module()


def completion(text):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
    )


def step(action="finish", arguments=None, thought="The answer is ready."):
    return (
        f"Thought: {thought}\nAction: {action}\n"
        f"Action Input: {json.dumps(arguments if arguments is not None else {})}"
    )


class RecordingTool(react.BaseTool):
    def __init__(self):
        super().__init__(name="lookup", description="Look up a query.")
        self.calls = []

    def get_schema(self):
        return {"function": {"name": self.name, "description": self.description}}

    def __call__(self, query):
        self.calls.append(query)
        if query == "raise":
            raise RuntimeError("backend unavailable")
        return f"Found: {query}"


class ParserTests(unittest.TestCase):
    def test_numbered_python_dict_and_crlf(self):
        parsed = react.parse_react_step(
            "Thought 2: Search now.\r\nAction 2: lookup\r\n"
            "Action Input 2: {'query': 'a {nested} phrase'}"
        )
        self.assertEqual(parsed.action_input, {"query": "a {nested} phrase"})

    def test_multiline_code_is_preserved(self):
        code = 'print("hello")\nprint({"key": 1})'
        parsed = react.parse_react_step(step("python_repl", {"code": code}))
        self.assertEqual(parsed.action_input["code"], code)

    def test_fenced_arguments_and_finish_without_arguments(self):
        parsed = react.parse_react_step(
            'Thought: Search.\nAction: lookup\nAction Input:\n```json\n{"query":"x"}\n```'
        )
        self.assertEqual(parsed.action_input, {"query": "x"})
        self.assertEqual(
            react.parse_react_step("Thought: Done.\nAction: Finish").action_input, {}
        )

    def test_invalid_or_executable_inputs_are_rejected(self):
        for raw in (
            '{"thought":"x","action":"finish","is_final":true}',
            "Thought: Search.\nAction: lookup",
            "Thought: Search.\nAction: lookup\nAction Input: ['x']",
            "Thought: Search.\nAction: lookup\nAction Input: {1: 'x'}",
            "Thought: Search.\nAction: lookup\nAction Input: dict(query='x')",
            step("lookup", {"query": "x"}) + "\nObservation: invented result",
            step("lookup", {"query": "x"}) + "\n" + step(),
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                react.parse_react_step(raw)


class AgentTests(unittest.TestCase):
    def make_agent(self, replies, tools=None, **kwargs):
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=Mock(side_effect=[completion(x) for x in replies]))
            )
        )
        with patch.object(react, "OpenAI", return_value=client):
            agent = react.ReActAgentNoSGR(
                model_name="test-model", tools=[] if tools is None else tools, **kwargs
            )
        return agent, client.chat.completions.create

    def test_tool_observation_and_separate_final_synthesis(self):
        tool = RecordingTool()
        agent, create = self.make_agent(
            [step("lookup", {"query": "pandas"}), step(), "result = 42"], tools=[tool]
        )
        self.assertEqual(agent.run("Solve the task"), "result = 42")
        self.assertEqual(tool.calls, ["pandas"])
        requests = [call.kwargs for call in create.call_args_list]
        self.assertEqual(len(requests), 3)
        for request in requests:
            self.assertEqual(set(request), {"model", "messages", "temperature"})
        self.assertIn("Observation from lookup: Found: pandas", str(requests[1]["messages"]))
        self.assertIn("TASK:\nSolve the task", requests[2]["messages"][1]["content"])

    def test_no_tools_never_creates_or_calls_default_tool(self):
        with patch.object(react, "LLMTool") as fallback:
            agent, create = self.make_agent([step("execute", {"code": "bad"}), step(), "answer"])
        fallback.assert_not_called()
        self.assertEqual(agent.tools, [])
        with self.assertLogs(level="INFO") as logs:
            self.assertEqual(agent.run("No tools"), "answer")
        self.assertTrue(any("AVAILABLE_TOOLS: []" in line for line in logs.output))
        self.assertFalse(any("TOOL_CALL:" in line for line in logs.output))
        self.assertIn("Unknown tool 'execute'", str(create.call_args_list[1].kwargs))

    def test_default_tool_only_when_tools_is_none(self):
        with patch.object(react, "OpenAI"), patch.object(react, "LLMTool", return_value=RecordingTool()) as fallback:
            agent = react.ReActAgentNoSGR(model_name="test", tools=None)
        fallback.assert_called_once()
        self.assertEqual(list(agent.tools_dict), ["lookup"])

    def test_parse_error_recovers_and_is_bounded(self):
        agent, create = self.make_agent(["not a step", step(), "answer"])
        self.assertEqual(agent.run("task"), "answer")
        self.assertIn("FORMAT ERROR", str(create.call_args_list[1].kwargs))
        agent, create = self.make_agent(["not a step"] * 5)
        self.assertIn("failed to produce a valid response", agent.run("task"))
        self.assertEqual(create.call_count, 5)

    def test_invalid_arguments_and_loop_do_not_execute_tool(self):
        tool = RecordingTool()
        agent, create = self.make_agent(
            [step("lookup", {"wrong": "x"}), step("lookup", {"query": "x"}),
             step("lookup", {"query": "x"}), step(), "answer"], tools=[tool]
        )
        self.assertEqual(agent.run("task"), "answer")
        self.assertEqual(tool.calls, ["x"])
        self.assertIn("VALIDATION ERROR", str(create.call_args_list[1].kwargs))
        self.assertIn("LOOP DETECTED", str(create.call_args_list[3].kwargs))

    def test_execution_error_becomes_observation(self):
        agent, create = self.make_agent(
            [step("lookup", {"query": "raise"}), step(), "answer"], tools=[RecordingTool()]
        )
        self.assertEqual(agent.run("task"), "answer")
        self.assertIn("EXECUTION ERROR in tool 'lookup'", str(create.call_args_list[1].kwargs))

    def test_max_iterations_and_history_window_match_source(self):
        agent, create = self.make_agent([step("lookup", {"query": str(i)}) for i in range(4)],
                                        tools=[RecordingTool()], max_iterations=4, history_context=2)
        self.assertIn("Maximum iterations", agent.run("original task"))
        messages = create.call_args_list[-1].kwargs["messages"]
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "original task")
        self.assertIn("Found: 2", messages[-1]["content"])

    def test_concurrent_runs_do_not_mix_tasks_or_histories(self):
        agent, create = self.make_agent([])
        barrier = Barrier(2)

        def respond(**kwargs):
            messages = kwargs["messages"]
            if messages[0]["content"].startswith("Output ONLY"):
                prompt = messages[1]["content"]
                task = "TASK_A" if "TASK:\nTASK_A" in prompt else "TASK_B"
                other = "TASK_B" if task == "TASK_A" else "TASK_A"
                self.assertNotIn(other, prompt)
                return completion(task)
            task = messages[1]["content"]
            barrier.wait(timeout=5)
            return completion(step(thought=task))

        create.side_effect = respond
        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertEqual(list(executor.map(agent.run, ["TASK_A", "TASK_B"])), ["TASK_A", "TASK_B"])
        self.assertEqual(agent.memory, [])

    def test_real_sdk_request_has_no_output_schema_or_function_calling(self):
        requests = []
        replies = iter([step(), "answer"])

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 0, "model": "test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": next(replies)},
                             "finish_reason": "stop"}],
            })

        with OpenAI(api_key="test", base_url="http://test/v1", http_client=httpx.Client(
            transport=httpx.MockTransport(handle)
        )) as client, patch.object(react, "OpenAI", return_value=client):
            agent = react.ReActAgentNoSGR(model_name="test-model", tools=[])
            self.assertEqual(agent.run("task"), "answer")
        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertEqual(set(request), {"model", "messages", "temperature"})


class ConfigTests(unittest.TestCase):
    def test_all_configs_load_and_react_modes_are_explicit(self):
        paths = list((ROOT / "pipeline_configs/react_no_sgr").rglob("*.yaml"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            with self.subTest(path=path.name):
                cfg = ConfigLoader().load_from_yaml(path)
                if cfg.type.name == "REACT":
                    self.assertEqual(cfg.components["agent"].type.name, "REACT_AGENT_NO_SGR")
                    self.assertEqual(cfg.components["agent"].params["few_shot_type"], "zero_shot")
                    self.assertIn("tools", cfg.components)

    def test_builder_injects_empty_tool_list(self):
        # Exercise the real registry/factory/builder, replacing only eager exports
        # of unrelated optional components with the real classes under test.
        name = "_react_pipeline_test"
        spec = importlib.util.spec_from_file_location(name, ROOT / "src/pipelines/templates/react_pipeline.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        agents = types.ModuleType("src.agents.pipelines")
        agents.ReActAgentNoSGR = react.ReActAgentNoSGR
        templates = types.ModuleType("src.pipelines.templates")
        templates.REACTPipeline = module.REACTPipeline
        cfg = ConfigLoader().load_from_yaml(ROOT / "pipeline_configs/react_no_sgr/baseline/react_no_tools.yaml")
        with patch.dict(sys.modules, {"src.agents.pipelines": agents, "src.pipelines.templates": templates}), \
             patch.dict(ComponentRegistry._registry, clear=True), \
             patch.object(react, "OpenAI"), patch.object(react, "LLMTool") as fallback:
            pipeline = PipelineBuilder().build(cfg)
        self.assertIsInstance(pipeline.agent, Agent)
        self.assertEqual(pipeline.agent.tools, [])
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
