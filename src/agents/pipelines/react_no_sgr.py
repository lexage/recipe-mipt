"""Text ReAct ablation of react_sgr.py, with the same loop and final synthesis.

Only static defaults and existing few-shot examples are reused from react_sgr.
No structured-output agent, response schema, or constrained decoding is used.
"""

import ast
import copy
import inspect
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.tools import BaseTool, LLMTool
from .react_sgr import AgentConfig, FEW_SHOT_REGISTRY, FINISH_PROMPT_TEMPLATE


_LOG_SEPARATOR = "\n" + "_" * 20 + "\n"

REACT_SYSTEM_PROMPT = """You are an autonomous ReAct agent.
Interact with tools to solve tasks. Minimize steps; finish immediately when the answer is known.

TOOLS (Case-Sensitive, Exact Names Only)
{tools_formatted}

Respond with ONE step in plain text:
Thought: Your reasoning for this step
Action: tool_name or finish
Action Input: {{"arg": "value"}}

CONSTRAINTS
1. Action MUST be one of: {tool_names} or 'finish'.
   IMPORTANT: Few-shot examples may demonstrate different tools for format/structure reference only. You must strictly call ONLY the available tools listed in {tool_names}.
2. Action Input contains the tool's named arguments as a JSON or Python dictionary.
3. If a tool error occurs, correct the arguments and retry once.
4. Do not generate observations or additional steps; wait for the tool result.

FINALIZATION
When the answer is known, use Action: finish and Action Input: {{}}.
"""

_STEP_PATTERN = re.compile(
    r"\A\s*Thought(?:\s+\d+)?:[ \t]*(?P<thought>.*?)"
    r"\r?\n[ \t]*Action(?:\s+\d+)?:[ \t]*(?P<action>[^\r\n]+)"
    r"(?:\r?\n[ \t]*Action Input(?:\s+\d+)?:[ \t]*(?P<arguments>.*))?\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class TextStep:
    """Local parser result; never sent to the model as an output schema."""

    thought: str
    action: str
    action_input: Dict[str, Any]
    raw_text: str


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```[^\n]*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def parse_react_step(text: str) -> TextStep:
    """Parse routing labels after free-text generation, without schema validation.

    Literal dictionaries are accepted to keep the existing few-shot examples
    usable. literal_eval never executes model-produced expressions.
    """
    match = _STEP_PATTERN.fullmatch(_strip_code_fence(text))
    if not match:
        raise ValueError("Expected Thought:, Action:, and Action Input: labels")

    thought = match.group("thought").strip()
    action = match.group("action").strip()
    arguments_text = match.group("arguments")
    if not thought or not action:
        raise ValueError("Thought and Action must be non-empty")

    if arguments_text is None:
        if action.lower() != "finish":
            raise ValueError("Action Input is required for a tool call")
        arguments = {}
    else:
        arguments_text = _strip_code_fence(arguments_text)
        try:
            arguments = json.loads(arguments_text)
        except json.JSONDecodeError:
            try:
                arguments = ast.literal_eval(arguments_text)
            except (ValueError, SyntaxError) as exc:
                raise ValueError("Action Input must be a literal dictionary") from exc
        if not isinstance(arguments, dict) or not all(
            isinstance(key, str) for key in arguments
        ):
            raise ValueError("Action Input must be a dictionary with string keys")

    return TextStep(thought, action, arguments, text)


class ReActAgentNoSGR(Agent):
    """ReAct with unrestricted text completions and locally parsed tool calls."""

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = AgentConfig.DEFAULT_TEMPERATURE,
        name: str = "ReActAgentNoSGR",
        instruction: Optional[str] = None,
        examples: Optional[list] = None,
        max_iterations: int = AgentConfig.DEFAULT_MAX_ITERATIONS,
        tools: Optional[List[BaseTool]] = None,
        history_context: int = AgentConfig.DEFAULT_HISTORY_CONTEXT,
        few_shot_type: str = "zero_shot",
    ):
        super().__init__(name)
        if not model_name:
            raise ValueError("model_name is required for LLM initialization")

        self.examples = examples or []
        self.max_iterations = max_iterations
        self.history_context = history_context
        # An explicitly empty list must stay empty for the no-tools baseline.
        self.tools = [LLMTool(url=url, model_name=model_name)] if tools is None else tools
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.tool_names = ", ".join(self.tools_dict)
        self.tools_prompt = "\n\n".join(
            tool.get_prompt_description() for tool in self.tools
        )
        self.instruction = instruction or REACT_SYSTEM_PROMPT.format(
            tool_names=self.tool_names or "(none)",
            tools_formatted=self.tools_prompt or "No tools are available.",
        )

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        if few_shot_type not in FEW_SHOT_REGISTRY:
            raise ValueError(
                f"Unknown few_shot_type: {few_shot_type}. "
                f"Available: {list(FEW_SHOT_REGISTRY)}"
            )
        self.few_shot_examples = FEW_SHOT_REGISTRY[few_shot_type]
        self._reset_runtime_state()

        logging.info("REACT_MODE: text; SGR: false; response_format: unset")
        logging.info("AVAILABLE_TOOLS: %s", list(self.tools_dict))
        logging.info("SYSTEM PROMPT: %s", self.instruction)
        logging.info(_LOG_SEPARATOR)

    def _reset_runtime_state(self) -> None:
        self._tool_call_retry_count = 0
        self.error_history: List[Dict[str, Any]] = []
        self._execution_history: List[Dict[str, Any]] = []
        self.memory: List[Dict[str, str]] = []

    def _get_text_response(self, messages: List[Dict[str, str]]) -> Optional[TextStep]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            text = response.choices[0].message.content or ""
            logging.info("RAW_RESPONSE: %s", text)
            return parse_react_step(text)
        except Exception as exc:
            logging.warning("REACT_RESPONSE_ERROR: %s: %s", type(exc).__name__, exc)
            return None

    def _get_sliding_window_messages(self) -> List[Dict[str, str]]:
        # Keep the same message-count window as react_sgr.py.
        if len(self.memory) <= AgentConfig.MIN_MEMORY_BASE_SIZE:
            return self.memory.copy()
        return self.memory[: AgentConfig.MIN_MEMORY_BASE_SIZE] + self.memory[
            AgentConfig.MIN_MEMORY_BASE_SIZE :
        ][-self.history_context :]

    def _validate_tool_args(
        self, tool_name: str, action_input: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        # Check only Python call compatibility, not a Pydantic response schema.
        try:
            inspect.signature(self.tools_dict[tool_name]).bind(**action_input)
        except TypeError as exc:
            return False, str(exc)
        return True, None

    def _detect_loop(self, action: str, action_input: Dict[str, Any]) -> bool:
        if len(self._execution_history) < AgentConfig.LOOP_DETECTION_WINDOW - 1:
            return False
        recent_calls = self._execution_history[
            -(AgentConfig.LOOP_DETECTION_WINDOW - 1) :
        ]
        return all(
            entry["action"] == action and entry["action_input"] == action_input
            for entry in recent_calls
        )

    def _format_error_observation(
        self, error_type: str, action: str = "", error_detail: str = ""
    ) -> str:
        error_templates = {
            "text_parse": (
                "FORMAT ERROR: Respond with ONE plain-text step using Thought:, "
                "Action:, and Action Input:. Action Input must be a dictionary; "
                "use Action: finish and Action Input: {} when done."
            ),
            "unknown_tool": (
                f"TOOL ERROR: Unknown tool '{action}'. "
                f"Available tools: {self.tool_names or '(none)'}. "
                "Tool names are CASE-SENSITIVE and must match exactly."
            ),
            "validation": (
                f"VALIDATION ERROR: {error_detail}. "
                "Ensure Action Input matches the arguments expected by the tool."
            ),
            "loop_detected": (
                f"LOOP DETECTED: Tool '{action}' was called 2+ times consecutively "
                "with identical arguments. Breaking execution cycle. "
                "Please try a different approach or tool."
            ),
            "execution": f"EXECUTION ERROR in tool '{action}': {error_detail}",
        }
        return error_templates.get(error_type, f"ERROR: {error_detail}")

    def _record_error(
        self, error_type: str, iteration: int, action: str = "", error_detail: str = ""
    ) -> None:
        if error_type != "loop_detected":
            self._tool_call_retry_count += 1
            self.error_history.append({"type": error_type, "step": iteration})
        if error_type == "text_parse":
            self.memory.append({"role": "assistant", "content": "[INVALID_RESPONSE]"})
        observation = self._format_error_observation(error_type, action, error_detail)
        self.memory.append({"role": "user", "content": f"Observation: {observation}"})
        logging.warning("OBSERVATION: %s", observation)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name not in self.tools_dict:
            return self._format_error_observation("unknown_tool", tool_name)
        logging.info("TOOL_CALL: %s", tool_name)
        try:
            result = self.tools_dict[tool_name](**arguments)
            return str(result) if result is not None else ""
        except Exception as exc:
            return self._format_error_observation(
                "execution", tool_name, f"{type(exc).__name__}: {exc}"
            )

    def _record_successful_execution(self, step: TextStep, observation: str) -> None:
        self._execution_history.append(
            {"action": step.action, "action_input": step.action_input}
        )
        if len(self._execution_history) > AgentConfig.MAX_EXECUTION_HISTORY_SIZE:
            self._execution_history.pop(0)
        self._tool_call_retry_count = 0
        self.memory.append(
            {"role": "user", "content": f"Observation from {step.action}: {observation}"}
        )
        logging.info("OBSERVATION: %s", observation)
        logging.info(_LOG_SEPARATOR)

    def _generate_final_answer(self, task: str) -> str:
        recent = self.memory[AgentConfig.MIN_MEMORY_BASE_SIZE :][
            -self.history_context * 2 :
        ]
        history_text = "\n\n".join(
            f"[{message['role'].upper()}]: {message['content']}" for message in recent
        )
        prompt = FINISH_PROMPT_TEMPLATE.format(task=task, history_text=history_text)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Output ONLY the final answer, no explanations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logging.warning("FINAL_ANSWER_ERROR: %s: %s", type(exc).__name__, exc)
            return "Error: Could not generate final answer."

    def run(self, task: str) -> str:
        # run_ds1000 shares one agent across worker threads. Mutable trajectory
        # state belongs to this call; configured tools and client are reused.
        return copy.copy(self)._run_task(task)

    def _run_task(self, task: str) -> str:
        self._reset_runtime_state()
        logging.info("TASK:\n%s", task)
        logging.info("AVAILABLE_TOOLS: %s", list(self.tools_dict))
        logging.info(_LOG_SEPARATOR)
        self.memory = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": self.few_shot_examples + task},
        ]

        for iteration in range(self.max_iterations):
            if self._tool_call_retry_count >= AgentConfig.MAX_RETRY_COUNT:
                logging.error("AGENT FAILED: Retry limit exceeded after %s iterations", iteration + 1)
                return (
                    "Error: Agent failed to produce a valid response after multiple attempts. "
                    "Please refine your request."
                )

            step = self._get_text_response(self._get_sliding_window_messages())
            if step is None:
                self._record_error("text_parse", iteration)
                continue

            logging.info("STEP %s:", iteration + 1)
            logging.info(_LOG_SEPARATOR)
            logging.info("THOUGHT: %s", step.thought)
            logging.info("ACTION: %s", step.action)
            logging.info("ACTION_INPUT: %s", step.action_input)
            logging.info(_LOG_SEPARATOR)
            self.memory.append({"role": "assistant", "content": step.raw_text})

            if step.action.lower() == "finish":
                logging.info("AGENT DECIDED TO FINISH at step %s", iteration + 1)
                final_answer = self._generate_final_answer(task)
                logging.info("FINAL ANSWER: %s", final_answer)
                logging.info(_LOG_SEPARATOR)
                return final_answer

            if step.action not in self.tools_dict:
                self._record_error("unknown_tool", iteration, step.action)
                continue
            valid, error = self._validate_tool_args(step.action, step.action_input)
            if not valid:
                self._record_error("validation", iteration, step.action, error)
                continue
            if self._detect_loop(step.action, step.action_input):
                self._record_error("loop_detected", iteration, step.action)
                continue

            observation = self.execute_tool(step.action, step.action_input)
            self._record_successful_execution(step, observation)

        return (
            "Error: Maximum iterations reached without finding a solution. "
            "Please refine your request or try a different approach."
        )
