import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union, TypedDict
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.tools import BaseTool, LLMTool


class AgentConfig:
    """Configuration constants for ReActAgent to avoid magic numbers."""

    MAX_RETRY_COUNT: int = 5
    MAX_EXECUTION_HISTORY_SIZE: int = 10
    MIN_MEMORY_BASE_SIZE: int = 2  # system + initial task messages
    LOOP_DETECTION_WINDOW: int = 2  # consecutive identical calls to trigger
    DEFAULT_TEMPERATURE: float = 0.0
    DEFAULT_MAX_ITERATIONS: int = 10
    DEFAULT_HISTORY_CONTEXT: int = 5
    MAX_LOGGED_RESPONSE_CHARS: int = 8_000


_LOG_SEPARATOR = "\n" + "_" * 20 + "\n"


def _normalize_openai_base_url(url: Optional[str]) -> str:
    """Return the API root expected by the OpenAI client.

    /v1/models is a resource endpoint used to list models, not a client
    base URL. Accepting it silently makes the SDK request
    /v1/models/chat/completions and previously produced only a generic
    retry-limit error because the HTTP exception was swallowed.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("url is required for LLM initialization")

    configured_url = url.strip().rstrip("/")
    normalized_url = configured_url
    if configured_url.endswith("/models"):
        normalized_url = configured_url[: -len("/models")].rstrip("/")
        if not normalized_url:
            raise ValueError("url must include an OpenAI-compatible API base")

        logging.warning(
            "LLM_BASE_URL_NORMALIZED\tconfigured=%s\teffective=%s\t"
            "reason=models_endpoint",
            configured_url,
            normalized_url,
        )

    return normalized_url


MessageDict = TypedDict("MessageDict", {"role": str, "content": str})


REACT_SYSTEM_PROMPT = """You are an autonomous ReAct agent. 
Interact with tools to solve tasks. Minimize steps; finish immediately when the answer is known.

TOOLS (Case-Sensitive, Exact Names Only)
{tools_formatted}

OUTPUT FORMAT (STRICT JSON)
Respond ONLY with a single valid JSON object matching this schema:
{{
  "thought": "Your reasoning for this step",
  "action": "tool_name" OR "finish",
  "action_input": {{ "arg": "value" }} (empty dict {{}} if action='finish'),
  "is_final": true (ONLY if action='finish')
}}

CONSTRAINTS
1. Action MUST be one of: {tool_names} or 'finish'.
2. action_input MUST match the tool's argument schema exactly.
3. If a tool error occurs, correct the arguments and retry once.

FINALIZATION
When action='finish':
- Set is_final=true.
"""

FINISH_PROMPT_TEMPLATE = """You are providing the FINAL ANSWER to the user's task.

TASK:
{task}

CONVERSATION HISTORY:
{history_text}

INSTRUCTIONS:
- Output ONLY the solution logic for TASK, no explanations"""


class AgentStep(BaseModel):
    """Structured output model for a single ReAct agent step."""

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(
        ..., description="Step-by-step reasoning about what to do next"
    )
    action: str = Field(..., description="Tool name to call, or 'finish' to complete")
    action_input: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool as key-value pairs"
    )
    is_final: bool = Field(
        default=False, description="Whether this is the final answer"
    )


class ReActAgentSGR(Agent):
    """ReAct agent for automatic programming tasks with structured JSON output."""

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = AgentConfig.DEFAULT_TEMPERATURE,
        name: str = "ReActAgent",
        instruction: Optional[str] = None,
        examples: Optional[list] = None,
        max_iterations: int = AgentConfig.DEFAULT_MAX_ITERATIONS,
        max_retries: int = AgentConfig.MAX_RETRY_COUNT,
        tools: Optional[List[BaseTool]] = None,
        history_context: int = AgentConfig.DEFAULT_HISTORY_CONTEXT,
    ):
        super().__init__(name)

        if not model_name:
            raise ValueError("model_name is required for LLM initialization")

        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        if history_context < 1:
            raise ValueError("history_context must be positive")

        self.examples = examples or []
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.history_context = history_context

        # Initialize tool registry with fallback to default LLMTool
        self.tools = tools or [LLMTool(url=url, model_name=model_name)]
        self.tools_dict = {t.name: t for t in self.tools}
        self.tool_names = ", ".join(self.tools_dict.keys())

        # Build tool descriptions for prompt injection
        self.tools_prompt = "\n\n".join(
            [t.get_prompt_description() for t in self.tools]
        )

        # Compose system prompt with dynamic tool information
        self.instruction = instruction or REACT_SYSTEM_PROMPT.format(
            tool_names=self.tool_names, tools_formatted=self.tools_prompt
        )

        # Initialize OpenAI client with vLLM-compatible configuration.
        # The client needs the API root (usually /v1), not /v1/models.
        self.base_url = _normalize_openai_base_url(url)
        self.client = OpenAI(base_url=self.base_url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature

        # Runtime state - reset on each run() call
        self._reset_runtime_state()

        # Memory stores conversation history for context window management
        # self.memory: List[MessageDict] = []

        logging.info(f"SYSTEM PROMPT: {self.instruction}")
        logging.info(_LOG_SEPARATOR)

    def _reset_runtime_state(self) -> None:
        """Reset ephemeral state between agent runs to prevent state leakage."""
        self._tool_call_retry_count = 0
        self._last_llm_error: Optional[str] = None
        self.error_history: List[Dict[str, Any]] = []
        self._execution_history: List[Dict[str, Any]] = []
        self.memory: List[MessageDict] = []

    @staticmethod
    def _bounded_log_text(value: Any) -> str:
        text = "" if value is None else str(value)
        limit = AgentConfig.MAX_LOGGED_RESPONSE_CHARS
        if len(text) <= limit:
            return text
        return text[:limit] + "\n[response truncated]"

    def _get_structured_response(
        self, messages: List[MessageDict], iteration: int
    ) -> Optional[AgentStep]:
        """Call the LLM, log the exchange and parse an AgentStep."""

        step = iteration + 1
        logging.info(
            "LLM_REQUEST\tstep=%s\tmodel=%s\tbase_url=%s\tmessages=%s",
            step,
            self.model_name,
            self.base_url,
            len(messages),
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format=AgentStep,
            )
            choice = response.choices[0]
            message = choice.message
            raw_content = self._bounded_log_text(message.content)
            logging.info(
                "LLM_RESPONSE\tstep=%s\tfinish_reason=%s\tcontent=%s",
                step,
                getattr(choice, "finish_reason", None),
                raw_content,
            )

            agent_step = message.parsed
            if agent_step is None:
                self._last_llm_error = (
                    "The completion returned no parsed AgentStep; "
                    f"raw content: {raw_content or '[empty]'}"
                )
                logging.error(
                    "LLM_RESPONSE_INVALID\tstep=%s\terror=%s",
                    step,
                    self._last_llm_error,
                )
                return None

            self._last_llm_error = None
            return agent_step

        except Exception as error:
            self._last_llm_error = f"{type(error).__name__}: {error}"
            logging.exception(
                "LLM_REQUEST_FAILED\tstep=%s\tmodel=%s\tbase_url=%s\terror=%s",
                step,
                self.model_name,
                self.base_url,
                self._last_llm_error,
            )
            return None

    def _get_sliding_window_messages(self) -> List[MessageDict]:
        """
        Get messages with sliding window to balance context vs token limits.

        Why keep first 2 messages: system prompt + initial task are essential
        for maintaining agent behavior and task grounding throughout execution.
        """
        if len(self.memory) <= AgentConfig.MIN_MEMORY_BASE_SIZE:
            return self.memory.copy()

        # Preserve system + task messages
        base_messages = self.memory[: AgentConfig.MIN_MEMORY_BASE_SIZE]

        # Take recent conversation turns within context window
        recent_messages = self.memory[AgentConfig.MIN_MEMORY_BASE_SIZE :][
            -self.history_context :
        ]

        return base_messages + recent_messages

    def _validate_tool_args(
        self, tool_name: str, action_input: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        tool = self.tools_dict.get(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found"
        if hasattr(tool, "arg_schema") and tool.arg_schema:
            try:
                tool.arg_schema(**action_input)
                return True, None
            except ValidationError as e:
                return False, f"Validation: {e}"
            except Exception as e:
                return False, f"Schema error: {type(e).__name__}: {e}"
        return isinstance(action_input, dict), "action_input must be a dict"

    def _detect_loop(self, action: str, action_input: Dict[str, Any]) -> bool:
        """Detect execution loops by checking for repeated identical tool calls."""
        if len(self._execution_history) < AgentConfig.LOOP_DETECTION_WINDOW - 1:
            return False

        # Check the last N-1 entries + current would make N total
        recent_calls = self._execution_history[
            -(AgentConfig.LOOP_DETECTION_WINDOW - 1) :
        ]
        return all(
            entry["action"] == action and entry["action_input"] == action_input
            for entry in recent_calls
        )

    def _format_error_observation(
        self,
        error_type: str,
        action: str = "",
        action_input: Union[Dict, str] = "",
        error_detail: str = "",
    ) -> str:
        """Format error messages for agent feedback using predefined templates."""
        available_tools = ", ".join(self.tools_dict.keys())

        error_templates = {
            "json_parse": (
                f"FORMAT ERROR: Invalid JSON response. "
                f"Please respond with a valid JSON object matching the AgentStep schema. "
                f"Error details: {error_detail}"
            ),
            "unknown_tool": (
                f"TOOL ERROR: Unknown tool '{action}'. "
                f"Available tools: {available_tools}. "
                f"Tool names are CASE-SENSITIVE and must match exactly."
            ),
            "validation": (
                f"VALIDATION ERROR: {error_detail}. "
                f"Ensure action_input matches the argument structure expected by the tool."
            ),
            "loop_detected": (
                f"LOOP DETECTED: Tool '{action}' was called 2+ times consecutively "
                f"with identical arguments. Breaking execution cycle. "
                f"Please try a different approach or tool."
            ),
            "execution": (f"EXECUTION ERROR in tool '{action}': {error_detail}"),
        }
        return error_templates.get(error_type, f"ERROR: {error_detail}")

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute tool and return observation or formatted error."""
        if tool_name not in self.tools_dict:
            return self._format_error_observation("unknown_tool", tool_name, arguments)

        tool = self.tools_dict[tool_name]
        try:
            result = tool(**arguments)
            return str(result) if result is not None else ""
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return self._format_error_observation(
                "execution", tool_name, arguments, error_msg
            )

    def _generate_final_answer(self, task: str) -> str:
        """Synthesize final answer from conversation history."""

        recent = self.memory[AgentConfig.MIN_MEMORY_BASE_SIZE :][
            -self.history_context * 2 :
        ]

        history_text = "\n\n".join(
            f"[{msg['role'].upper()}]: {msg['content']}" for msg in recent
        )

        prompt = FINISH_PROMPT_TEMPLATE.format(task=task, history_text=history_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Output ONLY the final answer, no explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "Error: Could not generate final answer."

    def _build_error_memory_entry(self, error_type: str, **kwargs) -> Dict[str, Any]:
        """Helper to create consistent error observation entries for memory."""
        return {
            "role": "user",
            "content": f"Observation: {self._format_error_observation(error_type, **kwargs)}",
        }

    def _handle_invalid_response(self, iteration: int) -> None:
        """Handle an HTTP, parsing or validation failure from the LLM."""
        self._tool_call_retry_count += 1
        error_detail = (
            self._last_llm_error or "Failed to parse or validate JSON response"
        )
        error_obs = self._format_error_observation(
            "json_parse", error_detail=error_detail
        )
        self.memory.append({"role": "assistant", "content": "[INVALID_RESPONSE]"})
        self.memory.append({"role": "user", "content": f"Observation: {error_obs}"})
        self.error_history.append(
            {"type": "json_parse", "step": iteration, "detail": error_detail}
        )
        logging.warning(
            "LLM_RETRY\tstep=%s\tretry=%s/%s\terror=%s",
            iteration + 1,
            self._tool_call_retry_count,
            self.max_retries,
            error_detail,
        )

    def _handle_unknown_tool(self, agent_step: AgentStep, iteration: int) -> None:
        """Handle reference to non-existent tool."""
        self._tool_call_retry_count += 1
        error_obs = self._format_error_observation(
            "unknown_tool", agent_step.action, agent_step.action_input
        )
        self.memory.append(
            self._build_error_memory_entry(
                "unknown_tool",
                action=agent_step.action,
                action_input=agent_step.action_input,
            )
        )
        self.error_history.append(
            {"type": "unknown_tool", "step": iteration, "tool": agent_step.action}
        )

    def _handle_validation_error(
        self, agent_step: AgentStep, validation_error: str, iteration: int
    ) -> None:
        """Handle tool argument validation failure."""
        self._tool_call_retry_count += 1
        error_obs = self._format_error_observation(
            "validation", agent_step.action, agent_step.action_input, validation_error
        )
        self.memory.append({"role": "user", "content": f"Observation: {error_obs}"})
        self.error_history.append({"type": "validation", "step": iteration})

    def _handle_loop_detection(self, agent_step: AgentStep) -> None:
        """Handle detected execution loop by breaking cycle."""
        error_obs = self._format_error_observation(
            "loop_detected", agent_step.action, agent_step.action_input
        )
        self.memory.append({"role": "user", "content": f"Observation: {error_obs}"})

    def _record_successful_execution(
        self, agent_step: AgentStep, observation: str
    ) -> None:
        """Update state after successful tool execution."""

        self._execution_history.append(
            {"action": agent_step.action, "action_input": agent_step.action_input}
        )
        if len(self._execution_history) > AgentConfig.MAX_EXECUTION_HISTORY_SIZE:
            self._execution_history.pop(0)

        self._tool_call_retry_count = 0

        self.memory.append(
            {
                "role": "user",
                "content": f"Observation from {agent_step.action}: {observation}",
            }
        )

        logging.info(f"OBSERVATION: {observation}")
        logging.info(_LOG_SEPARATOR)

    def run(self, task: str) -> str:
        """Execute ReAct loop to solve the programming task."""
        self._reset_runtime_state()

        logging.info(f"TASK:\n{task}")
        logging.info(_LOG_SEPARATOR)

        self.memory = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": task},
        ]

        for iteration in range(self.max_iterations):
            if self._tool_call_retry_count >= self.max_retries:
                last_error = (
                    self.error_history[-1].get("detail", "unknown LLM error")
                    if self.error_history
                    else "unknown LLM error"
                )
                logging.error(
                    "AGENT_FAILED\treason=retry_limit\tfailed_attempts=%s\t"
                    "last_error=%s",
                    self._tool_call_retry_count,
                    last_error,
                )
                return (
                    "Error: Agent failed to produce a valid structured response "
                    f"after {self._tool_call_retry_count} attempts. "
                    f"Last LLM error: {last_error}"
                )

            messages = self._get_sliding_window_messages()

            agent_step = self._get_structured_response(messages, iteration)

            if agent_step is None:
                self._handle_invalid_response(iteration)
                continue

            logging.info(f"STEP {iteration+1}:")
            logging.info(_LOG_SEPARATOR)
            logging.info(f"THOUGHT: {agent_step.thought}")
            logging.info(f"ACTION: {agent_step.action}")
            logging.info(f"ACTION_INPUT: {agent_step.action_input}")
            logging.info(f"IS_FINAL: {agent_step.is_final}")
            logging.info(_LOG_SEPARATOR)

            self.memory.append(
                {
                    "role": "assistant",
                    "content": json.dumps(agent_step.model_dump(), ensure_ascii=False),
                }
            )

            if agent_step.is_final or agent_step.action.lower().strip() == "finish":
                logging.info(f"AGENT DECIDED TO FINISH at step {iteration+1}")
                logging.info(_LOG_SEPARATOR)

                final_answer = self._generate_final_answer(task)

                logging.info(f"FINAL ANSWER: {final_answer}")
                logging.info(_LOG_SEPARATOR)
                return final_answer

            if agent_step.action not in self.tools_dict:
                self._handle_unknown_tool(agent_step, iteration)
                continue

            is_valid, validation_error = self._validate_tool_args(
                agent_step.action, agent_step.action_input
            )
            if not is_valid:
                self._handle_validation_error(agent_step, validation_error, iteration)
                continue

            if self._detect_loop(agent_step.action, agent_step.action_input):
                self._handle_loop_detection(agent_step)
                continue

            observation = self.execute_tool(agent_step.action, agent_step.action_input)
            self._record_successful_execution(agent_step, observation)

        return (
            "Error: Maximum iterations reached without finding a solution. "
            "Please refine your request or try a different approach."
        )