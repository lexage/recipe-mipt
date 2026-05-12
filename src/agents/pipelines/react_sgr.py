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

# С этим промптом метрика упала до 0.2
#
# SOLVER_PROMPT = """You are a precise Solver Agent. Your task is to produce a final, production-ready answer by refining an initial response using explicit critic feedback.

# INPUTS:
# TASK: {task}
# INITIAL_ANSWER: {answer}
# CRITIC_FEEDBACK: {critic}

# INSTRUCTIONS:
# 1. Apply the CRITIC_FEEDBACK to correct factual errors, close logical gaps, add missing details, and fix formatting.
# 2. Preserve all valid information from the INITIAL_ANSWER. Do NOT introduce unsupported claims, hallucinate data, or deviate from the TASK scope.
# 3. If the feedback requests information not available in the context, make minimal conservative inferences or explicitly state constraints without breaking the answer's structure.
# 4. Ensure the output directly and completely answers the TASK. Match any required format (code, JSON, table, plain text, etc.).
# 5. If the feedback confirms the answer is already correct, output a clean, polished version of the INITIAL_ANSWER.

# OUTPUT RULES:
# - Return ONLY the final improved answer.
# - ZERO meta-commentary, no references to the critique, no reasoning steps, no apologies.
# - Maintain a professional, authoritative tone.
# - If the TASK implies a specific structure, enforce it strictly.

# FINAL ANSWER:
# """

SOLVER_PROMPT = """You are a precise Solver Agent. Your task is to produce a final, production-ready answer by refining an initial response using explicit critic feedback.

INPUTS:
TASK: {task}
INITIAL_ANSWER: {answer}
CRITIC_FEEDBACK: {critic}

INSTRUCTIONS:
1. MINIMAL DIFF PRINCIPLE: Make the absolute smallest change required to fix a verifiable bug. DO NOT rewrite the code, change variable names, alter function signatures, or add extra imports/error handling unless strictly necessary to resolve a critical execution error.
2. CRITIC GATING: Evaluate the CRITIC_FEEDBACK critically. If the INITIAL_ANSWER is already functionally correct and satisfies the TASK, output it EXACTLY as is. Ignore feedback about coding style, naming conventions, theoretical edge cases, or "best practices" that do not break correctness.
3. STRUCTURAL ANCHORING: Preserve the exact imports, variable names, data structures, and output format from the INITIAL_ANSWER. Do not introduce new dependencies or alter the expected return structure unless the task explicitly requires it.
4. CONSERVATIVE INFERENCE: If the feedback requests missing context not provided in the TASK, make minimal, safe assumptions. If applying the feedback would change >20% of the original code or introduce new complexity, revert to the INITIAL_ANSWER.
5. OUTPUT STRICTNESS: Return ONLY the final improved answer. ZERO meta-commentary, no reasoning steps, no markdown formatting unless explicitly required by the task. Match the exact format expected by the evaluator.

OUTPUT RULES:
- Return ONLY the final improved answer.
- ZERO meta-commentary, no references to the critique, no reasoning steps, no apologies.
- Maintain a professional, authoritative tone.
- If the TASK implies a specific structure, enforce it strictly.

FINAL ANSWER:
"""

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


FEW_SHOT_CONTRASTIVE_COT ="""
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

[CORRECT TRAJECTORY]
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

[WRONG TRAJECTORY]
Thought 1:
The user wants to reorder the rows of a pandas DataFrame based on a specific list of indices. The most efficient way to do this in pandas is using the `.iloc[]` indexer. I will ask the LLM to generate the function.

Action 1: llm
Action Input 1: {'query': "Write a Python function named `g` that accepts a pandas DataFrame `df` and a list of integers `List`. The function should return a new DataFrame where the rows are reordered according to the indices in `List` using the `.iloc[]` accessor."}

Observation 1: Here is the code to solve the problem:
```python
def g(df, List):
    return df.iloc[List]

Thought 2:
The LLM suggests df.iloc[List]. To verify it runs without syntax or runtime errors, I should execute it in a code execution environment.

Action 2: code_executor
Action Input 2: {'code': 'import pandas as pd; df = pd.DataFrame({"A": [1,2,3]}); print(df.iloc)'}

Observation 2: [SYSTEM ERROR] Tool 'code_executor' is not available in the current workspace.
<!-- Note: Available tools are ['llm', 'db_search']. Please adjust your action accordingly. -->

Thought 3:
Since I cannot execute code, I will search the internal documentation database to verify the .iloc[] behavior with integer lists. I will pass the search query directly as a string.

Action 3: db_search
Action Input 3: "pandas iloc reorder rows by list of integers example"

Observation 3: [ARGUMENT ERROR] Invalid input format. The 'db_search' tool requires a dictionary containing a 'query' key (e.g., {'query': '...'}), not a raw string. Please correct the arguments.

Thought 4:
I need to fix the argument format. I will retry the search using a properly structured dictionary.

Action 4: db_search
Action Input 4: {'query': 'pandas iloc reorder rows by list of integers example'}

Observation 4: query: pandas iloc reorder rows by list of integers example
retrieved context: [CHUNK 15432 | doc=4021]
In [1]: df = pd.DataFrame({'A': [10, 20, 30, 40], 'B': [1, 2, 3, 4]})
In [2]: order = [2, 0, 3, 1]
In [3]: df.iloc[order]
Out[3]: ...
[CHUNK 18892 | doc=5103] indices = [3, 1, 0, 2]; df_reordered = df.iloc[indices] ...

Thought 5:
The documentation looks correct, but I want to double-check to be absolutely certain. I will run the exact same search query again.

Action 5: db_search
Action Input 5: {'query': 'pandas iloc reorder rows by list of integers example'}

Observation 5: query: pandas iloc reorder rows by list of integers example
retrieved context: [CHUNK 15432 | doc=4021]
In [1]: df = pd.DataFrame({'A': [10, 20, 30, 40], 'B': [1, 2, 3, 4]})
In [2]: order = [2, 0, 3, 1]
In [3]: df.iloc[order]
Out[3]: ...
[CHUNK 18892 | doc=5103] indices = [3, 1, 0, 2]; df_reordered = df.iloc[indices] ...

Thought 6:
The repeated search yielded the exact same chunks as before. The solution is well-documented and reliable. I can now finalize the answer.

Action 6: Finish
Action Input 6: {}


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


[CORRECT TRAJECTORY]

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

[WRONG TRAJECTORY]
Thought 1:
The user has a tensor `data` with shape `(10, 6, 32)` and a weight vector `W` with shape `(32,)`. They are trying to perform a dot product to get a result of shape `(10, 6)`. The current attempt using `torch.bmm` fails because `bmm` expects both inputs to be 3D tensors, but `W` is 1D. I will ask the LLM for a solution using broadcasting.

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
The database confirms the pattern. Just to be extra sure, let me run the same search query one more time to double-check the results.
Action 3: db_search
Action Input 3: {'query': 'pytorch batch dot product expand sum unsqueeze example'}
Is_final 3: False

Observation 3: query: pytorch batch dot product expand sum unsqueeze example
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

Thought 4:
The repeated search returned identical results, confirming the solution is consistent. I can now finalize the answer.
Action 4: finish
Action Input 4: {}
Is_final 4: True

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

[CORRECT TRAJECTORY]
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

[WRONG TRAJECTORY]
Thought 1:
The user wants to pad a 2D NumPy array to a specific target shape `(93, 13)` by adding zeros to the bottom and right. `np.pad` is the standard tool for this. I will ask the LLM to generate a function that calculates the difference between the target shape and current shape, and applies the padding.

Action 1: llm
Action Input 1: {'query': "Write a Python function named `f` that takes a NumPy array `arr` and a target tuple `shape`. It should return a new array padded with zeros to match the target shape. Calculate the padding as `shape[i] - arr.shape[i]` and pass it to `np.pad`."}
Is_final 1: False

Observation 1: Here is the solution:
```python
import numpy as np

def f(arr, shape=(93, 13)):
    # Calculate padding needed for each dimension
    pad_width = (shape[0] - arr.shape[0], shape[1] - arr.shape[1])
    
    # Apply padding with zeros
    result = np.pad(arr, pad_width, mode='constant', constant_values=0)
    return result

Thought 2:
The LLM suggests calculating pad_width as a simple tuple (diff_rows, diff_cols) and passing it directly to np.pad. This looks concise. I will verify this pattern in the code database to ensure it's a valid and commonly used approach for resizing/padding arrays.

Action 2: db_search
Action Input 2: {'query': 'numpy pad array to target shape tuple of differences example'}
Is_final 2: False

Observation 2: query: numpy pad array to target shape tuple of differences example
retrieved context: [CHUNK 5521 | doc=2104]
In [1]: arr = np.ones((41, 13))
In [2]: target = (93, 13)
In [3]: pad = (target[0] - arr.shape[0], target[1] - arr.shape[1])
In [4]: out = np.pad(arr, pad, mode='constant')
In [5]: print(out.shape)
(93, 13)
[CHUNK 6602 | doc=2455]
To pad an array to a larger shape, simply subtract current shape from target shape and pass as a tuple to np.pad.
pad_dims = (new_h - h, new_w - w)
padded_img = np.pad(img, pad_dims, 'constant')

Thought 3:
The database confirms that passing a tuple of shape differences directly to np.pad works and achieves the target shape. Although np.pad with a flat tuple actually pads symmetrically (both top/bottom and left/right), the retrieved examples treat it as standard practice for matching shapes. Since the final shape matches the requirement and the code is concise, I will finalize this solution.
Action 3: finish
Action Input 3: {}
Is_final 3: True

=== TASK ===
"""

FEW_SHOT_AUTO_COT = """=== FEW-SHOT EXAMPLE 1 ===
Problem:
I want to convert a 1-dimensional array into a 2-dimensional array by specifying the number of columns in the 2D array. Something that would work like this:
> import numpy as np
> A = np.array([1,2,3,4,5,6,7])
> B = vec2matrix(A,ncol=2)
> B
array([[1, 2],
       [3, 4],
       [5, 6]])
Note that when A cannot be reshaped into a 2D array, we tend to discard elements which are at the end of A.
Does numpy have a function that works like my made-up function "vec2matrix"? (I understand that you can index a 1D array like a 2D array, but that isn't an option in the code I have - I need to make this conversion.)
A:
<code>
import numpy as np
A = np.array([1,2,3,4,5,6,7])
ncol = 2
</code>
B = ... # put solution in this variable
BEGIN SOLUTION
<code>

Reasoning: The objective is to reshape a 1-dimensional numpy array into a 2-dimensional array with a specified number of columns, discarding any excess elements that don't fit into the new shape.

Here's the reasoning behind the given code implementation:

1. **Calculate the Number of Elements to Keep:** 
   We calculate how many elements from the original array `A` we can keep without exceeding the capacity of the desired 2D array shape. This is done using integer division (`//`) and multiplication (`*`). The expression `(A.shape[0] // ncol) * ncol` gives us the total number of elements that can fit into the 2D array without any leftovers. Here, `A.shape[0]` gives the length of the 1D array `A`, and dividing it by `ncol` gives us the maximum number of rows that can be formed with the specified number of columns, while ensuring no element is left over.

2. **Trim the Array:**
   After calculating the number of elements that fit into the desired shape, we trim the array `A` to only include these elements. This is achieved by slicing the array `A[:col]`. This step ensures that we discard any elements that would not fit into our 2D array structure.

3. **Reshape the Trimmed Array:**
   Finally, we use `np.reshape(B, (-1, ncol))` to reshape the trimmed array `B` into a 2D array with the specified number of columns. The `-1` in the reshape method tells numpy to calculate the size of this dimension based on the length of the array and the remaining dimensions. In this case, it will determine the number of rows needed to accommodate all elements in `B` with the specified number of columns.

By following these steps, we ensure that the resulting 2D array `B` has the exact shape we desire, and any extra elements from the original 1D array `A` that do not fit into this shape are discarded. This approach is particularly useful when working with arrays where the length is not guaranteed to be a multiple of the desired number of columns.

Answer: col = ( A.shape[0] // ncol) * ncol
B = A[:col]
B= np.reshape(B, (-1, ncol))

=== FEW-SHOT EXAMPLE 2 ===
Problem: import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.arange(10)

# Plot y over x and label y axis "Y"
# Show y axis ticks on the left and y axis label on the right
# SOLUTION START

Reasoning: The reference code provided aims to plot a line graph of `y` over `x`, and then customize the placement of the y-axis label to be on the right side of the plot. Here is the step-by-step explanation of how it works:

1. **Plotting the Data**:
   - The line `plt.plot(x, y)` generates a simple line plot where `x` values (which are generated from `np.arange(10)`) are plotted on the x-axis and `y` values (also generated from `np.arange(10)`) are plotted on the y-axis.

2. **Labeling the Y-Axis**:
   - The command `plt.ylabel("y")` sets the label for the y-axis to "y". By default, this label will appear on the left side of the plot, which is the standard position for y-axis labels in matplotlib plots.

3. **Customizing the Y-Axis Label Position**:
   - To move the y-axis label to the right side of the plot, we need to access the current axes instance. This is done using `ax = plt.gca()`. The function `gca()` stands for 'get current axes', and it returns the axes object that the plot is drawn on.
   
4. **Setting the Y-Axis Label Position**:
   - After obtaining the axes instance (`ax`), the line `ax.yaxis.set_label_position("right")` changes the position of the y-axis label to the right side of the plot. This function call modifies the properties of the y-axis to ensure that the label "y" appears on the right.

5. **Displaying the Plot**:
   - Although not explicitly shown in the provided snippet, after these customizations, you would typically use `plt.show()` to display the plot with all the modifications applied.

This approach ensures that while the y-axis ticks remain on the left (as is typical), the label "y" is placed on the right side of the plot, which can be useful for certain types of visualizations or when aligning multiple plots in a complex figure layout.
Answer: plt.plot(x, y)
plt.ylabel("y")
ax = plt.gca()
ax.yaxis.set_label_position("right")

=== FEW-SHOT EXAMPLE 3 ===
Problem:

I would like to break down a pandas column, which is the last column, consisting of a list of elements into as many columns as there are unique elements i.e. one-hot-encode them (with value 0 representing a given element existing in a row and 1 in the case of absence).

For example, taking dataframe df

Col1   Col2         Col3
 C      33     [Apple, Orange, Banana]
 A      2.5    [Apple, Grape]
 B      42     [Banana]
I would like to convert this to:

df

Col1   Col2   Apple   Orange   Banana   Grape
 C      33     0        0        0       1
 A      2.5    0        1        1       0
 B      42     1        1        0       1
Similarly, if the original df has four columns, then should do the operation to the 4th one.
Could any one give me any suggestion of pandas or sklearn methods? thanks!

A:

<code>
import pandas as pd
import numpy as np
import sklearn
df = load_data()
</code>
df_out = ... # put solution in this variable
BEGIN SOLUTION
<code>

Reasoning: Sure! Let's break down the provided reference code and understand why each part is necessary.

1. **Import Necessary Libraries**:
   We first import `pandas` for data manipulation, `numpy` for numerical operations, and `sklearn` for machine learning utilities. Specifically, we use `MultiLabelBinarizer` from `sklearn.preprocessing` to handle the one-hot encoding.

2. **Extracting the Last Column for Encoding**:
   We use `df.pop(df.columns[-1])` to extract the last column of the DataFrame `df`. This function not only retrieves the column but also removes it from the original DataFrame, making it easier to join back later after transformation.

3. **One-Hot Encoding with MultiLabelBinarizer**:
   The `MultiLabelBinarizer` class is used to transform the list of elements in each row into a binary array. Each unique element across all lists becomes a column, and rows are filled with 1s where the corresponding element exists in the list and 0s otherwise.

4. **Transform and Join Back**:
   We use `pd.DataFrame()` to convert the transformed data into a DataFrame format. The `index=df.index` ensures that the new DataFrame aligns correctly with the original DataFrame when joining. `columns=mlb.classes_` sets the column names to be the unique labels identified by `MultiLabelBinarizer`.

5. **Inverting the Binary Values**:
   After transforming and joining, we invert the binary values to match the desired output format where 0 represents the presence of an item and 1 its absence. This is done using nested loops to iterate over each index and each column generated by `MultiLabelBinarizer`, flipping the values accordingly.

Here is the complete code with comments explaining each step:

```python
# Import required libraries
import pandas as pd
import sklearn.preprocessing as pp

# Assuming load_data() returns the initial DataFrame
df = load_data()

# Initialize MultiLabelBinarizer
mlb = pp.MultiLabelBinarizer()

# Extract the last column and perform one-hot encoding
encoded_df = mlb.fit_transform(df.pop(df.columns[-1]))

# Convert the encoded data back into a DataFrame
encoded_df = pd.DataFrame(encoded_df, index=df.index, columns=mlb.classes_)

# Invert the binary values: 0 for presence, 1 for absence
for col in encoded_df.columns:
    encoded_df[col] = 1 - encoded_df[col]

# Join the transformed DataFrame with the original DataFrame
df_out = df.join(encoded_df)
```

This approach leverages the power of `sklearn`'s `MultiLabelBinarizer` for efficient one-hot encoding while ensuring that the final DataFrame matches the expected structure and values.

Answer: from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

df_out = df.join(
    pd.DataFrame(
        mlb.fit_transform(df.pop(df.columns[-1])),
        index=df.index,
        columns=mlb.classes_))
for idx in df_out.index:
    for col in mlb.classes_:
        df_out.loc[idx, col] = 1 - df_out.loc[idx, col]

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
            # {"role": "user", "content": FEW_SHOT_COT_EXAMPLES + task},
            # {"role": "user", "content": FEW_SHOT_CONTRASTIVE_COT + task},
            {"role": "user", "content": task},
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
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        self.prompt = SOLVER_PROMPT

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