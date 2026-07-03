import logging
import json
import importlib
from typing import List, Dict, Any, Optional, Tuple, Union, TypedDict
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.tools import BaseTool, LLMTool


_PROMPTS_PACKAGE = "src.agents.pipelines.prompts"


def _load_react_sgr_prompts(dataset: str):
    """Load ReActAgentSGR prompt templates for the given dataset.

    Each dataset package (e.g. ``ds1000``, ``codemmlu``) exposes the SAME public
    names (REACT_SYSTEM_PROMPT, FINISH_PROMPT_TEMPLATE, SOLVER_PROMPT,
    FEW_SHOT_REGISTRY), so switching benchmarks only changes which module we
    import — selected from the pipeline config via the ``dataset`` param.
    """
    module = importlib.import_module(f"{_PROMPTS_PACKAGE}.{dataset}.prompts_react_sgr")
    return (
        module.REACT_SYSTEM_PROMPT,
        module.FINISH_PROMPT_TEMPLATE,
        module.SOLVER_PROMPT,
        module.FEW_SHOT_REGISTRY,
    )


# CodeMMLU is evaluated on two sub-splits (code_completion / fill_in_the_middle) by a
# SINGLE pipeline instance (see codemmlu_agent_pipelines.py). The runner prepends a
# per-kind steering HINT to every task, and these stable phrases let the agent detect
# which sub-split the live task belongs to so it can load the matching few-shot split
# (e.g. `cot_code` vs `cot_middle`) instead of one combined string. On datasets without
# sub-splits (e.g. DS-1000) no marker matches and the agent keeps its base few-shots.
_SUBSET_TASK_MARKERS = (
    ("code", "REAL bugs"),
    ("middle", "functionally EQUIVALENT"),
)


def _detect_subset(task) -> Optional[str]:
    """Return the CodeMMLU sub-split ('code' / 'middle') for a task, or None."""
    text = task if isinstance(task, str) else str(task)
    for subset, marker in _SUBSET_TASK_MARKERS:
        if marker in text:
            return subset
    return None


def _resolve_subset_few_shots(
    few_shot_type: str, registry: Dict[str, str]
) -> Dict[Optional[str], str]:
    """Map each CodeMMLU sub-split to the few-shot string it should use.

    From a base ``few_shot_type`` (e.g. ``cot``) look up the per-sub-split variants
    ``{base}_code`` / ``{base}_middle`` in the registry. Whichever exist are used for
    that sub-split; any missing one — and the ``None`` (undetected) case — falls back to
    the base string. So CodeMMLU gets ``cot_code``/``cot_middle`` automatically, while a
    dataset without those keys (DS-1000) or an explicit ``*_code``/``zero_shot`` type
    keeps the previous single-few-shot behaviour unchanged.
    """
    base = registry[few_shot_type]
    return {
        "code": registry.get(f"{few_shot_type}_code", base),
        "middle": registry.get(f"{few_shot_type}_middle", base),
        None: base,
    }


class AgentConfig:
    """Configuration constants for ReActAgent to avoid magic numbers."""

    MAX_RETRY_COUNT: int = 5
    MAX_EXECUTION_HISTORY_SIZE: int = 10
    MIN_MEMORY_BASE_SIZE: int = 2  # system + initial task messages
    LOOP_DETECTION_WINDOW: int = 2  # consecutive identical calls to trigger
    DEFAULT_TEMPERATURE: float = 0.0
    DEFAULT_MAX_ITERATIONS: int = 10
    DEFAULT_HISTORY_CONTEXT: int = 5


_LOG_SEPARATOR = "\n" + "_" * 20 + "\n"

MessageDict = TypedDict("MessageDict", {"role": str, "content": str})


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
        tools: Optional[List[BaseTool]] = None,
        history_context: int = AgentConfig.DEFAULT_HISTORY_CONTEXT,
        few_shot_type: str = "zero_shot",
        dataset: str = "ds1000",
    ):
        super().__init__(name)

        if not model_name:
            raise ValueError("model_name is required for LLM initialization")

        # Dataset selects which prompt package to use (ds1000, codemmlu, ...).
        react_system_prompt, finish_prompt_template, _, few_shot_registry = (
            _load_react_sgr_prompts(dataset)
        )
        self.dataset = dataset
        self.finish_prompt_template = finish_prompt_template

        self.examples = examples or []
        self.max_iterations = max_iterations
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
        self.instruction = instruction or react_system_prompt.format(
            tool_names=self.tool_names, tools_formatted=self.tools_prompt
        )

        # Initialize OpenAI client with vLLM-compatible configuration
        self.client = OpenAI(base_url=url, api_key="vllm", timeout=600.0, max_retries=2)
        self.model_name = model_name
        self.temperature = temperature

        if few_shot_type not in few_shot_registry:
            raise ValueError(f"Unknown few_shot_type: {few_shot_type}. Available: {list(few_shot_registry.keys())}")

        # Resolve a per-sub-split few-shot mapping so the code/middle CodeMMLU splits
        # each get their own examples (e.g. cot_code / cot_middle), selected per run().
        self._few_shots_by_subset = _resolve_subset_few_shots(
            few_shot_type, few_shot_registry
        )
        self.few_shot_examples = few_shot_registry[few_shot_type]

        # Runtime state - reset on each run() call
        self._reset_runtime_state()

        # Memory stores conversation history for context window management
        # self.memory: List[MessageDict] = []

        logging.info(f"SYSTEM PROMPT: {self.instruction}")
        logging.info(_LOG_SEPARATOR)

    def _select_few_shots(self, task: str) -> str:
        """Pick the few-shot split matching this task's CodeMMLU sub-split."""
        subset = _detect_subset(task)
        return self._few_shots_by_subset.get(subset, self.few_shot_examples)

    def _reset_runtime_state(self) -> None:
        """Reset ephemeral state between agent runs to prevent state leakage."""
        self._tool_call_retry_count = 0
        self.error_history: List[Dict[str, Any]] = []
        self._execution_history: List[Dict[str, Any]] = []
        self.memory: List[MessageDict] = []

    def _get_structured_response(
        self, messages: List[MessageDict]
    ) -> Optional[AgentStep]:
        """Call LLM with JSON output constraint and parse response via Pydantic."""
        try:
            response = self.client.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format=AgentStep,
            )

            agent_step = response.choices[0].message.parsed

            if agent_step is None:
                return None

            return agent_step

        except Exception as e:
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

        prompt = self.finish_prompt_template.format(task=task, history_text=history_text)

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
        """Handle case when LLM response cannot be parsed as AgentStep."""
        self._tool_call_retry_count += 1
        error_obs = self._format_error_observation(
            "json_parse", error_detail="Failed to parse or validate JSON response"
        )
        self.memory.append({"role": "assistant", "content": "[INVALID_RESPONSE]"})
        self.memory.append({"role": "user", "content": f"Observation: {error_obs}"})
        self.error_history.append({"type": "json_parse", "step": iteration})

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
            # {"role": "user", "content": FEW_SHOT_COT_EXAMPLES + task},
            # {"role": "user", "content": FEW_SHOT_CONTRASTIVE_COT + task},
            {"role": "user", "content": self._select_few_shots(task) + task},
            # {"role": "user", "content": task},
        ]

        for iteration in range(self.max_iterations):
            if self._tool_call_retry_count >= AgentConfig.MAX_RETRY_COUNT:
                logging.error(
                    f"AGENT FAILED: Retry limit exceeded after {iteration+1} iterations"
                )
                return (
                    "Error: Agent failed to produce a valid response after multiple attempts. "
                    "Please refine your request."
                )

            messages = self._get_sliding_window_messages()

            agent_step = self._get_structured_response(messages)

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
        
class SolverReAct(Agent):
    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        name: str = "ReActSolverAgent",
        temperature: float = 0.0,
        dataset: str = "ds1000",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm", timeout=600.0, max_retries=2)
        self.model_name = model_name
        self.temperature = temperature
        self.dataset = dataset
        _, _, solver_prompt, _ = _load_react_sgr_prompts(dataset)
        self.prompt = solver_prompt

    def llm(self, prompt: str) -> str:
        """Сalling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def run(self, task, react_answer, critic_answer):

        solve_prompt = self.prompt.format(answer=react_answer, critic=critic_answer, task=task)
        final_answer = self.llm(solve_prompt).strip()
        logging.info(f"SOLVE PROMPT:\n\n{solve_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"FINAL ANSWER SOLVER:\n\n{final_answer}")
        logging.info(_LOG_SEPARATOR)
        return final_answer