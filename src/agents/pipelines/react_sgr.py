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


_LOG_SEPARATOR = "\n" + "_" * 20 + "\n"

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
   IMPORTANT: Few-shot examples may demonstrate different tools for format/structure reference only. You must strictly call ONLY the available tools listed in {tool_names}.
2. action_input MUST match the tool's argument schema exactly.
3. If a tool error occurs, correct the arguments and retry once.

FINALIZATION
When action='finish':
- Set is_final=true.
"""

# FINISH_PROMPT_TEMPLATE = """You are providing the FINAL ANSWER to the user's task.

# TASK:
# {task}

# CONVERSATION HISTORY:
# {history_text}

# INSTRUCTIONS:
# - Output ONLY the solution logic for TASK, no explanations"""


FINISH_PROMPT_TEMPLATE = """You are providing the FINAL ANSWER to the user's task.

CONVERSATION HISTORY:
{history_text}

TASK:
{task}

INSTRUCTIONS:
- Output ONLY the solution logic for TASK, no explanations"""


FEW_SHOT_COT_EXAMPLES = """
=== FEW-SHOT EXAMPLE 1 ===
TASK:
Problem:
I have the following DataFrame:
    Col1  Col2  Col3  Type
0      1     2     3     1
1      4     5     6     1
2      7     8     9     2
3    10    11    12     2
4    13    14    15     3
5    16    17    18     3

The DataFrame is read from a CSV file. All rows which have Type 1 are on top, followed by the rows with Type 2, followed by the rows with Type 3, etc.
I would like to shuffle the order of the DataFrame's rows according to a list.
For example, give a list [2, 4, 0, 3, 1, 5] and desired result should be:
    Col1  Col2  Col3  Type
2      7     8     9     2
4     13    14    15     3
0     1     2     3     1
3    10    11    12     2
1     4     5     6     1
5    16    17    18     3
...
How can I achieve this?

A:
<code>
import pandas as pd
import numpy as np

df = pd.DataFrame({'Col1': [1, 4, 7, 10, 13, 16],
                   'Col2': [2, 5, 8, 11, 14, 17],
                   'Col3': [3, 6, 9, 12, 15, 18],
                   'Type': [1, 1, 2, 2, 3, 3]})
List = np.random.permutation(len(df))
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Thought 1:
The user wants to reorder the rows of a pandas DataFrame based on a specific list of indices. The most efficient way to do this in pandas is using the `.iloc[]` indexer, which selects rows by integer location. I will ask the LLM to generate the function.

Action 1: llm
Action Input 1: {'query': "Write a Python function named `g` that accepts a pandas DataFrame `df` and a list of integers `List`. The function should return a new DataFrame where the rows are reordered according to the indices in `List` using the `.iloc[]` accessor."}

Observation 1: Here is the code to solve the problem:

```python
def g(df, List):
    return df.iloc[List]
```

Thought 2:
The LLM suggests df.iloc[List]. To verify the exact behavior with integer lists and ensure there are no index alignment side effects, I will search the code database for concrete examples.

Action 2: db_search
Action Input 2: {'query': 'pandas iloc reorder rows by list of integers example'}

Observation 2: query: pandas iloc reorder rows by list of integers example
retrieved context: [CHUNK 15432 | doc=4021]
In [1]: df = pd.DataFrame({'A': [10, 20, 30, 40], 'B': [1, 2, 3, 4]})
In [2]: order = [2, 0, 3, 1]
In [3]: df.iloc[order]
Out[3]:
A B
2 30 3
0 10 1
3 40 4
1 20 2

[CHUNK 18892 | doc=5103]
indices = [3, 1, 0, 2]
df_reordered = df.iloc[indices]
print(df_reordered)
Col1 Col2
3 40 4
1 20 2
0 10 1
2 30 3

Thought 3:
The retrieved code snippets confirm that df.iloc[list_of_indices] correctly reorders rows by positional integer selection without modifying the original index labels. The solution is validated and ready.

Action 3: Finish
Action Input 3: {}

=== FEW-SHOT EXAMPLE 2 ===
TASK:
Problem:


I have batch data and want to dot() to the data. W is trainable parameters. How to dot between batch data and weights?
Here is my code below, how to fix it?


hid_dim = 32
data = torch.randn(10, 2, 3, hid_dim)
data = data.view(10, 2*3, hid_dim)
W = torch.randn(hid_dim) # assume trainable parameters via nn.Parameter
result = torch.bmm(data, W).squeeze() # error, want (N, 6)
result = result.view(10, 2, 3)




A:


corrected, runnable code
<code>
import numpy as np
import pandas as pd
import torch
hid_dim = 32
data = torch.randn(10, 2, 3, hid_dim)
data = data.view(10, 2 * 3, hid_dim)
W = torch.randn(hid_dim)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>




Thought 1:
The user has a tensor `data` with shape `(10, 6, 32)` and a weight vector `W` with shape `(32,)`. They are trying to perform a dot product to get a result of shape `(10, 6)`. The current attempt using `torch.bmm` fails because `bmm` expects both inputs to be 3D tensors (batch matrix multiplication), but `W` is 1D. To fix this, I need to either expand `W` to match the dimensions of `data` for element-wise multiplication followed by summation, or unsqueeze `W` to make it a valid matrix for `bmm`. A common efficient way in PyTorch for this specific case (vector-matrix multiplication across a batch) is to use broadcasting: expand `W` to `(1, 6, 32)` (or match `data`'s shape exactly) and sum over the last dimension.


Action 1: llm
Action Input 1: {'query': "Write Python code using PyTorch to fix the following issue: 'data' has shape (10, 6, 32) and 'W' has shape (32,). Calculate the dot product such that the result has shape (10, 6). Use broadcasting (unsqueeze/expand) and sum, rather than bmm, to handle the dimensions correctly."}
Is_final 1: False

Observation 1: Here is the corrected code snippet:


```python
# Expand W to match the dimensions of data: (1, 1, 32) -> (10, 6, 32)
# Then multiply element-wise and sum over the last dimension (dim=2)
W_expanded = W.unsqueeze(0).unsqueeze(0).expand(*data.size())
result = torch.sum(data * W_expanded, dim=2)


# Reshape back to (10, 2, 3) as requested
result = result.view(10, 2, 3)

Thought 2:
The LLM suggests using unsqueeze, expand, and sum. Before finalizing, I should verify this pattern in the code database to ensure it's a commonly used approach for batch dot products in PyTorch.

Action 2: db_search
Action Input 2: {'query': 'pytorch batch dot product expand sum unsqueeze example'}
Is_final 2: False

Observation 2: query: pytorch batch dot product expand sum unsqueeze example
retrieved context: [CHUNK 7721 | doc=2901]
In [1]: data = torch.randn(10, 6, 32)
In [2]: W = torch.randn(32)
In [3]: W_exp = W.unsqueeze(0).unsqueeze(0).expand_as(data)
In [4]: out = (data * W_exp).sum(dim=2)
In [5]: out.shape
Out[5]: torch.Size([10, 6])

[CHUNK 9154 | doc=3387]
batch = torch.randn(32, 10, 64)
weights = torch.randn(64)
Expand weights and compute dot product
w_exp = weights.unsqueeze(0).unsqueeze(0).expand(*batch.size())
result = torch.sum(batch * w_exp, dim=-1)
print(result.shape)
torch.Size([32, 10])


Thought 3:
The database confirms that the unsqueeze + expand + sum pattern is a valid and commonly used approach for batch dot products in PyTorch. The retrieved examples match the LLM's suggestion exactly. I will now format this as the final answer.

Action 3: finish
Action Input 3: {}
Is_final 3: True

=== FEW-SHOT EXAMPLE 3 ===
TASK:
Problem:
I have a file with arrays or different shapes. I want to zeropad all the array to match the largest shape. The largest shape is (93,13).
To test this I have the following code:
arr = np.ones((41,13))
how can I zero pad this array to match the shape of (93,13)? And ultimately, how can I do it for thousands of rows? Specifically, I want to pad to the right and bottom of original array in 2D.




A:
<code>
import numpy as np
example_arr = np.ones((41, 13))
def f(arr = example_arr, shape=(93,13)):
# return the solution in this function
# result = f(arr, shape=(93,13))
### BEGIN SOLUTION


Thought 1:
The user wants to pad a 2D NumPy array to a specific target shape `(93, 13)` by adding zeros to the bottom and right. The input array `arr` has shape `(41, 13)`. The standard way to do this in NumPy is using `np.pad`. I need to calculate the padding width for each dimension. For the first dimension (rows), we need `93 - 41 = 52` rows at the bottom. For the second dimension (columns), we need `13 - 13 = 0` columns. The padding specification for `np.pad` is a tuple of tuples: `((top, bottom), (left, right))`. So the argument should be `((0, 52), (0, 0))`. I will ask the LLM to generate the function that implements this logic dynamically based on the target shape.


Action 1: llm
Action Input 1: {'query': "Write a Python function named `f` that takes a NumPy array `arr` and a target tuple `shape`. It should return a new array padded with zeros to match the target shape, padding only at the bottom and right sides. Use `np.pad`."}
Is_final 1: False



Observation 1: Here is the solution:


```python
import numpy as np


def f(arr, shape=(93, 13)):
    # Calculate padding needed for each dimension
    # Format: ((before_1, after_1), (before_2, after_2), ...)
    pad_width = (
        (0, shape[0] - arr.shape[0]), # Pad 0 at top, remainder at bottom
        (0, shape[1] - arr.shape[1])  # Pad 0 at left, remainder at right
    )
    
    # Apply padding
    result = np.pad(arr, pad_width, mode='constant', constant_values=0)
    return result

Thought 2:
The LLM suggests using np.pad with dynamic padding width calculation. Before finalizing, I should verify this pattern in the code database to ensure it's the standard approach for zero-padding arrays to a target shape in NumPy.

Action 2: db_search
Action Input 2: {'query': 'numpy pad array to target shape zeros bottom right example'}
Is_final 2: False

Observation 2: query: numpy pad array to target shape zeros bottom right example
retrieved context: [CHUNK 4521 | doc=1802]
In [1]: arr = np.ones((41, 13))
In [2]: target_shape = (93, 13)
In [3]: pad_width = ((0, target_shape[0] - arr.shape[0]), (0, target_shape[1] - arr.shape[1]))
In [4]: padded = np.pad(arr, pad_width, mode='constant')
In [5]: padded.shape
Out[5]: (93, 13)

[CHUNK 7834 | doc=2915]
arr = np.random.rand(5, 5)
pad = ((0, 10-arr.shape[0]), (0, 10-arr.shape[1]))
result = np.pad(arr, pad, mode='constant')
print(result.shape)
(10, 10)

Thought 3:
The database confirms that np.pad with dynamic padding width calculation ((0, shape[0]-arr.shape[0]), (0, shape[1]-arr.shape[1])) is the standard approach for zero-padding arrays to a target shape. The retrieved examples match the LLM's suggestion exactly. I will now format this as the final answer.

Action 3: finish
Action Input 3: {}
Is_final 3: True

=== TASK ===
"""

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
    ):
        super().__init__(name)

        if not model_name:
            raise ValueError("model_name is required for LLM initialization")

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
        self.instruction = instruction or REACT_SYSTEM_PROMPT.format(
            tool_names=self.tool_names, tools_formatted=self.tools_prompt
        )

        # Initialize OpenAI client with vLLM-compatible configuration
        self.client = OpenAI(base_url=url, api_key="vllm")
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
            {"role": "user", "content": FEW_SHOT_COT_EXAMPLES + task},
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