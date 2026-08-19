import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union, TypedDict
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.tools import BaseTool, LLMTool, ToolResult


class AgentConfig:
    """Configuration constants for ReActAgent to avoid magic numbers."""

    MAX_RETRY_COUNT: int = 5
    MAX_EXECUTION_HISTORY_SIZE: int = 10
    MIN_MEMORY_BASE_SIZE: int = 2  # system + initial task messages
    LOOP_DETECTION_WINDOW: int = 3  # initial call + one corrected retry
    DEFAULT_TEMPERATURE: float = 0.0
    DEFAULT_MAX_ITERATIONS: int = 10
    DEFAULT_HISTORY_CONTEXT: int = 5
    MAX_LOGGED_RESPONSE_CHARS: int = 8_000
    MAX_CONSECUTIVE_TOOL_FAILURES: int = 2


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
Use the available tools iteratively to solve the task. Finish only when repository
evidence and validation support the solution.

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
3. Treat tool observations as evidence. A successful call with no new information
   is not progress.
4. If a tool error is retryable, correct the arguments and retry at most once.
   If it is not retryable, or the same strategy keeps failing, change strategy.
5. Repository paths passed to tools are relative to the task repository. Never
   prefix them with /testbed.
6. Choose tools only from the dynamically generated list above and follow each
   selected tool's description. A successful edit proves only that it was applied;
   inspect the resulting repository changes and validate behavior.

FINALIZATION
When action='finish':
- Set is_final=true.
- Set action_input to an empty object.
- Use available tools to review the current repository changes before finishing.
  Run a relevant validation command after the most recent edit when an appropriate
  tool is available. If validation cannot run, inspect the failure and avoid
  unrelated changes.
"""

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

FEW_SHOT_AUTO_COT = """=== FEW-SHOT EXAMPLE 1===\nTASK:\nProblem: I want to convert a 1-dimensional array into a 2-dimensional array by specifying the number of columns in the 2D array. Something that would work like this:\n> import numpy as np\n> A = np.array([1,2,3,4,5,6,7])\n> B = vec2matrix(A,ncol=2)\n> B\narray([[1, 2],\n       [3, 4],\n       [5, 6]])\nNote that when A cannot be reshaped into a 2D array, we tend to discard elements which are at the end of A.\nDoes numpy have a function that works like my made-up function \"vec2matrix\"? (I understand that you can index a 1D array like a 2D array, but that isn't an option in the code I have - I need to make this conversion.)\n\nA:\n<code>\nimport numpy as np\nA = np.array([1,2,3,4,5,6,7])\nncol = 2\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\nThought 1: The numpy library does not have a direct function named \"vec2matrix\". However, we can achieve the desired functionality by first trimming the array to a size that fits the specified number of columns and then reshaping it.\nAction 1: llm\nAction Input 1: {'query': 'How do I reshape a 1D numpy array into a 2D array while discarding extra elements if necessary?'}\nIs_final 1: False\n\nObservation 1: To reshape a 1D array into a 2D array with a specific number of columns, you can first calculate the largest multiple of the column count that is less than or equal to the length of the array. Then, slice the array up to this point and reshape it using `numpy.reshape`.\n\nThought 2: Let's implement the logic based on the observation to trim the array and reshape it.\nAction 2: llm\nAction Input 2: {'query': 'Implement the logic to reshape a 1D numpy array into a 2D array with a given number of columns, discarding extra elements if necessary.'}\nIs_final 2: False\n\nObservation 2: Here is the implementation based on the previous discussion:\n```python\ncol = (A.shape[0] // ncol) * ncol\ntrimmed_A = A[:col]\nresult = trimmed_A.reshape(-1, ncol)\n```\n\nThought 3: The implementation has been completed and the solution is ready.\nAction 3: Finish\nAction Input 3: {}\nIs_final 3: True\n</code>\nAnswer: col = ( A.shape[0] // ncol) * ncol\nB = A[:col]\nB= np.reshape(B, (-1, ncol))\n\n\n=== FEW-SHOT EXAMPLE 2===\nTASK:\nProblem: import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nx = np.arange(10)\ny = np.arange(10)\n\n# Plot y over x and label y axis \"Y\"\n# Show y axis ticks on the left and y axis label on the right\n# SOLUTION START\n\nA:\n<code>\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nx = np.arange(10)\ny = np.arange(10)\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\nThought 1: To plot y over x and set up the y-axis label on the right side, we need to adjust the current plotting configuration.\nAction 1: llm\nAction Input 1: {'query': 'Plot y over x, label y axis \"Y\", show y axis ticks on the left and y axis label on the right'}\nIs_final 1: False\n\nObservation 1: The task requires modifying the current plotting setup to ensure the y-axis label appears on the right while keeping the y-axis ticks on the left. This can be achieved by adjusting the axis properties.\n\nThought 2: Proceeding with the plotting and customizing the y-axis settings.\nAction 2: llm\nAction Input 2: {'query': 'Adjust y-axis settings to show ticks on the left and label on the right'}\nIs_final 2: False\n\nObservation 2: Adjusting the y-axis to display the label on the right side while keeping the ticks on the left involves setting the label position property of the y-axis to 'right'.\n\nThought 3: Finalize the solution by implementing the plotting and customization.\nAction 3: Finish\nAction Input 3: {}\nIs_final 3: True\n</code>\nAnswer: plt.plot(x, y)\nplt.ylabel(\"y\")\nax = plt.gca()\nax.yaxis.set_label_position(\"right\")\n\n=== FEW-SHOT EXAMPLE 3===\nTASK:\nProblem: I would like to break down a pandas column, which is the last column, consisting of a list of elements into as many columns as there are unique elements i.e. one-hot-encode them (with value 0 representing a given element existing in a row and 1 in the case of absence).\n\nA:\n<code>\nfrom sklearn.preprocessing import MultiLabelBinarizer\nimport pandas as pd\n</code>\ndf_out = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\nThought 1: The problem requires transforming the last column of a DataFrame containing lists of items into multiple columns, effectively one-hot encoding those items. The values should be flipped so that presence is marked with 0 and absence with 1.\nAction 1: llm\nAction Input 1: {'query': 'Transform the last column of a DataFrame into multiple one-hot encoded columns, but flip the binary values so that presence is marked with 0 and absence with 1.'}\nIs_final 1: False\n\nObservation 1: \nTo achieve the transformation described, we can use `sklearn.preprocessing.MultiLabelBinarizer` to first one-hot encode the lists in the last column. Then, we can join these new columns back to the original DataFrame. Finally, we will invert the binary values to match the desired output format.\n\nThought 2: Implement the described logic using MultiLabelBinarizer and DataFrame operations.\nAction 2: llm\nAction Input 2: {'query': 'Implement the logic using MultiLabelBinarizer and DataFrame operations.'}\n\nObservation 2: \n```python\nmlb = MultiLabelBinarizer()\n# Transform the last column into a one-hot encoded format\none_hot_encoded = mlb.fit_transform(df[df.columns[-1]])\n# Create a DataFrame from the transformed data\nencoded_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)\n# Join the new DataFrame with the original, excluding the last column\ndf_out = df.iloc[:, :-1].join(encoded_df)\n# Flip the binary values so that presence is 0 and absence is 1\ndf_out[df.columns[-1]] = df[df.columns[-1]]\ndf_out[mlb.classes_] = 1 - df_out[mlb.classes_]\n```\n\nThought 3: The solution is ready, with the DataFrame `df_out` containing the transformed columns as required.\nAction 3: Finish\nAction Input 3: {}\nIs_final 3: True\n</code>\nAnswer: from sklearn.preprocessing import MultiLabelBinarizer\n\nmlb = MultiLabelBinarizer()\n\ndf_out = df.join(\n    pd.DataFrame(\n        mlb.fit_transform(df.pop(df.columns[-1])),\n        index=df.index,\n        columns=mlb.classes_))\nfor idx in df_out.index:\n    for col in mlb.classes_:\n        df_out.loc[idx, col] = 1 - df_out.loc[idx, col]\n\n"""

FEW_SHOT_REGISTRY = {
    "zero_shot": "",
    "cot": FEW_SHOT_COT_EXAMPLES,
    "contrastive_cot": FEW_SHOT_CONTRASTIVE_COT,
    "auto_cot": FEW_SHOT_AUTO_COT,
}

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

    @model_validator(mode="before")
    @classmethod
    def normalize_final_state(cls, data: Any) -> Any:
        """Derive redundant finalization fields from the selected action.

        Some OpenAI-compatible servers occasionally emit a correct tool action
        with a stale ``is_final`` value. Rejecting the whole response discards a
        useful action, so the action remains the single source of truth.
        """

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        action = normalized.get("action")
        if isinstance(action, str):
            is_finish = action.strip().lower() == "finish"
            normalized["is_final"] = is_finish
            if is_finish:
                normalized["action_input"] = {}
        return normalized


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
        max_tokens: Optional[int] = None,
        synthesize_final_answer: bool = True,
        require_repository_change_before_finish: bool = False,
        require_diff_before_finish: bool = False,
        require_validation_before_finish: bool = False,
    ):
        super().__init__(name)

        if not model_name:
            raise ValueError("model_name is required for LLM initialization")

        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if history_context < 1:
            raise ValueError("history_context must be positive")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if few_shot_type not in FEW_SHOT_REGISTRY:
            raise ValueError(
                f"Unknown few_shot_type: {few_shot_type}. "
                f"Available: {list(FEW_SHOT_REGISTRY)}"
            )

        self.examples = examples or []
        self.max_iterations = max_iterations
        self.history_context = history_context
        self.max_tokens = max_tokens
        self.synthesize_final_answer = synthesize_final_answer
        self.require_repository_change_before_finish = (
            require_repository_change_before_finish
        )
        self.require_diff_before_finish = require_diff_before_finish
        self.require_validation_before_finish = require_validation_before_finish
        self.few_shot_examples = FEW_SHOT_REGISTRY[few_shot_type]

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
        self._last_llm_error_code: Optional[str] = None
        self.error_history: List[Dict[str, Any]] = []
        self._execution_history: List[Dict[str, Any]] = []
        self._blocked_call_signatures: set[str] = set()
        self._failure_counts: Dict[Tuple[str, str, str, int], int] = {}
        self._last_failure_signature: Optional[Tuple[str, str, str, int]] = None
        self._consecutive_tool_failures = 0
        self._evidence_signatures: set[str] = set()
        self._repository_revision = 0
        self._last_diff_revision = -1
        self._last_validation_revision = -1
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
            "LLM_REQUEST\tstep=%s\tmodel=%s\tbase_url=%s\tmessages=%s\t"
            "max_tokens=%s",
            step,
            self.model_name,
            self.base_url,
            len(messages),
            self.max_tokens,
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format=AgentStep,
                **(
                    {"max_tokens": self.max_tokens}
                    if self.max_tokens is not None
                    else {}
                ),
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
                self._last_llm_error_code = "invalid_response"
                logging.error(
                    "LLM_RESPONSE_INVALID\tstep=%s\terror=%s",
                    step,
                    self._last_llm_error,
                )
                return None

            self._last_llm_error = None
            self._last_llm_error_code = None
            return agent_step

        except Exception as error:
            self._last_llm_error = f"{type(error).__name__}: {error}"
            self._last_llm_error_code = (
                "response_too_long"
                if type(error).__name__ == "LengthFinishReasonError"
                else "request_failed"
            )
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
        Keep the configured number of complete action/observation turns.

        Why keep first 2 messages: system prompt + initial task are essential
        for maintaining agent behavior and task grounding throughout execution.
        Each ReAct turn contributes an assistant action and a user observation,
        so history_context counts turns rather than individual messages.
        """
        if len(self.memory) <= AgentConfig.MIN_MEMORY_BASE_SIZE:
            return self.memory.copy()

        # Preserve system + task messages
        base_messages = self.memory[: AgentConfig.MIN_MEMORY_BASE_SIZE]

        # Take complete recent action/observation pairs within the context window.
        recent_messages = self.memory[AgentConfig.MIN_MEMORY_BASE_SIZE :][
            -self.history_context * 2 :
        ]

        return base_messages + recent_messages

    def _validate_tool_args(
        self, tool_name: str, action_input: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        tool = self.tools_dict.get(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found"
        if hasattr(tool, "validate_arguments"):
            try:
                validation_error = tool.validate_arguments(action_input)
            except Exception as e:
                return False, f"Schema error: {type(e).__name__}: {e}"
            return validation_error is None, validation_error
        if hasattr(tool, "arg_schema") and tool.arg_schema:
            try:
                tool.arg_schema(**action_input)
                return True, None
            except ValidationError as e:
                return False, f"Validation: {e}"
            except Exception as e:
                return False, f"Schema error: {type(e).__name__}: {e}"
        return isinstance(action_input, dict), "action_input must be a dict"

    @staticmethod
    def _normalized_action_input(
        action: str, action_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize arguments that can change without changing the strategy."""

        normalized = dict(action_input)
        if action == "apply_patch" and isinstance(normalized.get("patch"), str):
            normalized["patch"] = re.sub(
                r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@",
                "@@ <location> @@",
                normalized["patch"],
                flags=re.MULTILINE,
            )
        return normalized

    def _call_signature(self, action: str, action_input: Dict[str, Any]) -> str:
        normalized = self._normalized_action_input(action, action_input)
        encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        return f"{self._repository_revision}:{action}:{encoded}"

    @staticmethod
    def _normalized_failure_detail(detail: str) -> str:
        """Remove volatile identifiers from otherwise equivalent failures."""

        normalized = re.sub(r"agent-[0-9a-f]+\.patch", "agent.patch", detail)
        normalized = re.sub(r"\b\d+\b", "#", normalized)
        return " ".join(normalized.split())

    def _detect_loop(self, action: str, action_input: Dict[str, Any]) -> bool:
        """Detect repeated strategies even when other read calls intervene."""

        signature = self._call_signature(action, action_input)
        previous_calls = sum(
            entry.get("call_signature") == signature
            for entry in self._execution_history
        )
        return previous_calls >= AgentConfig.LOOP_DETECTION_WINDOW - 1

    def _is_meaningful_progress(
        self, agent_step: AgentStep, result: ToolResult
    ) -> bool:
        """Distinguish successful execution from new evidence or state change."""

        if result.progress is not None:
            return result.progress
        observation = str(result).strip()
        if not observation:
            return False
        evidence_signature = self._call_signature(
            agent_step.action, agent_step.action_input
        ) + ":" + observation
        if evidence_signature in self._evidence_signatures:
            return False
        self._evidence_signatures.add(evidence_signature)
        return True

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
                f"LOOP DETECTED: Tool '{action}' repeated the same strategy "
                f"without repository progress. Breaking execution cycle. "
                f"Change the arguments, inspect different evidence, or use another tool."
            ),
            "execution": (f"EXECUTION ERROR in tool '{action}': {error_detail}"),
        }
        return error_templates.get(error_type, f"ERROR: {error_detail}")

    def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        """Execute any tool and preserve its generic success or error status."""
        if tool_name not in self.tools_dict:
            return ToolResult.error(
                self._format_error_observation(
                    "unknown_tool", tool_name, arguments
                ),
                error_code="unknown_tool",
                retryable=False,
            )

        tool = self.tools_dict[tool_name]
        try:
            result = tool(**arguments)
            if isinstance(result, ToolResult):
                return result
            return ToolResult.ok(str(result) if result is not None else "")
        except Exception as error:
            error_msg = f"{type(error).__name__}: {error}"
            return ToolResult.error(
                self._format_error_observation(
                    "execution", tool_name, arguments, error_msg
                ),
                error_code="execution_error",
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
                **(
                    {"max_tokens": self.max_tokens}
                    if self.max_tokens is not None
                    else {}
                ),
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
        """Preserve baseline retry input while logging the concrete LLM error."""
        self._tool_call_retry_count += 1
        model_error_detail = "Failed to parse or validate JSON response"
        if self._last_llm_error_code == "response_too_long":
            model_error_detail = (
                "The previous response exceeded the model output limit. Return one "
                "short JSON step with a concise thought and exactly one tool call. "
                "Do not repeat file contents; split investigation and edits across "
                "multiple steps."
            )
        diagnostic_detail = self._last_llm_error or model_error_detail
        error_obs = self._format_error_observation(
            "json_parse", error_detail=model_error_detail
        )
        self.memory.append({"role": "assistant", "content": "[INVALID_RESPONSE]"})
        self.memory.append({"role": "user", "content": f"Observation: {error_obs}"})
        self.error_history.append(
            {
                "type": "json_parse",
                "step": iteration,
                "detail": diagnostic_detail,
            }
        )
        logging.warning(
            "LLM_RETRY\tstep=%s\tretry=%s/%s\terror=%s",
            iteration + 1,
            self._tool_call_retry_count,
            AgentConfig.MAX_RETRY_COUNT,
            diagnostic_detail,
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
        """Block one repeated call while allowing a genuinely changed strategy."""

        self._blocked_call_signatures.add(
            self._call_signature(agent_step.action, agent_step.action_input)
        )
        self._tool_call_retry_count = 0
        error_obs = self._format_error_observation(
            "loop_detected", agent_step.action, agent_step.action_input
        )
        self.memory.append(
            {
                "role": "user",
                "content": (
                    f"Observation: {error_obs} This exact call is blocked. "
                    "A changed input or different tool remains available."
                ),
            }
        )

    def _handle_blocked_call(self, agent_step: AgentStep) -> None:
        self._tool_call_retry_count = 0
        self.memory.append(
            {
                "role": "user",
                "content": (
                    f"Observation: This exact '{agent_step.action}' call is blocked "
                    "after a non-retryable or repeated failure. Change the input "
                    "or use a different strategy."
                ),
            }
        )

    def _record_successful_execution(
        self, agent_step: AgentStep, result: ToolResult
    ) -> None:
        """Record successful execution without equating success with progress."""

        observation = str(result)
        made_progress = self._is_meaningful_progress(agent_step, result)
        call_signature = self._call_signature(
            agent_step.action, agent_step.action_input
        )
        self._execution_history.append(
            {
                "action": agent_step.action,
                "action_input": agent_step.action_input,
                "call_signature": call_signature,
                "success": True,
                "progress": made_progress,
            }
        )
        if len(self._execution_history) > AgentConfig.MAX_EXECUTION_HISTORY_SIZE:
            self._execution_history.pop(0)

        self._tool_call_retry_count = 0
        if made_progress:
            self._last_failure_signature = None
            self._consecutive_tool_failures = 0

        tool = self.tools_dict.get(agent_step.action)
        if made_progress and getattr(tool, "mutates_repository", False):
            self._repository_revision += 1
        if getattr(tool, "provides_diff", False):
            self._last_diff_revision = self._repository_revision
        if result.validation_status == "passed":
            self._last_validation_revision = self._repository_revision

        self.memory.append(
            {
                "role": "user",
                "content": (
                    f"Observation from {agent_step.action} "
                    f"[progress={str(made_progress).lower()}]: {observation}"
                ),
            }
        )

        logging.info(
            "OBSERVATION\ttool=%s\tprogress=%s\tcontent=%s",
            agent_step.action,
            str(made_progress).lower(),
            observation,
        )
        logging.info(_LOG_SEPARATOR)

    def _finish_guard_error(self) -> Optional[str]:
        """Return actionable missing evidence for the current repository revision."""

        missing = []
        if self.require_repository_change_before_finish and self._repository_revision == 0:
            missing.append("apply a repository change with an editing tool")
        if (
            self.require_diff_before_finish
            and self._last_diff_revision != self._repository_revision
        ):
            missing.append("review the current full diff with git_diff")
        if (
            self.require_validation_before_finish
            and self._last_validation_revision != self._repository_revision
        ):
            missing.append(
                "run a successful relevant test/build/check command after the last edit "
                "(use git diff --check only as a fallback when project tests cannot run)"
            )
        if not missing:
            return None
        return "FINISH BLOCKED: Before finishing, " + "; then ".join(missing) + "."

    def _record_failed_execution(
        self, agent_step: AgentStep, result: ToolResult
    ) -> None:
        """Record a failed tool call without treating it as progress."""

        error_code = result.error_code or "tool_error"
        detail_fingerprint = self._normalized_failure_detail(str(result))
        signature = (
            agent_step.action,
            error_code,
            detail_fingerprint,
            self._repository_revision,
        )
        self._failure_counts[signature] = self._failure_counts.get(signature, 0) + 1
        if signature == self._last_failure_signature:
            self._consecutive_tool_failures += 1
        else:
            self._last_failure_signature = signature
            self._consecutive_tool_failures = 1

        self._execution_history.append(
            {
                "action": agent_step.action,
                "action_input": agent_step.action_input,
                "call_signature": self._call_signature(
                    agent_step.action, agent_step.action_input
                ),
                "success": False,
                "progress": False,
                "error_code": error_code,
            }
        )
        if len(self._execution_history) > AgentConfig.MAX_EXECUTION_HISTORY_SIZE:
            self._execution_history.pop(0)

        call_is_blocked = (
            not result.retryable
            or self._failure_counts[signature]
            >= AgentConfig.MAX_CONSECUTIVE_TOOL_FAILURES
        )
        if call_is_blocked:
            self._blocked_call_signatures.add(
                self._call_signature(agent_step.action, agent_step.action_input)
            )

        self._tool_call_retry_count = 0
        self.error_history.append(
            {
                "type": "tool_execution",
                "tool": agent_step.action,
                "detail": str(result),
                "error_code": error_code,
            }
        )
        blocked_note = (
            " This exact call is blocked; change its input or strategy."
            if call_is_blocked
            else " Correct the input before the single allowed retry."
            if result.retryable
            else " This failure is not retryable; change strategy."
        )
        observation = (
            f"TOOL FAILED [error_code={error_code}, "
            f"retryable={str(result.retryable).lower()}]: {result}{blocked_note}"
        )
        self.memory.append(
            {
                "role": "user",
                "content": f"Observation from {agent_step.action}: {observation}",
            }
        )
        logging.warning("OBSERVATION: %s", observation)
        logging.info(_LOG_SEPARATOR)

    def run(self, task: str) -> str:
        """Execute ReAct loop to solve the programming task."""
        self._reset_runtime_state()

        logging.info(f"TASK:\n{task}")
        logging.info(_LOG_SEPARATOR)

        self.memory = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": self.few_shot_examples + task},
        ]

        for iteration in range(self.max_iterations):
            if self._tool_call_retry_count >= AgentConfig.MAX_RETRY_COUNT:
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
                    "Error: Agent failed to produce a valid response after "
                    "multiple attempts. Please refine your request."
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

            if agent_step.action.lower().strip() == "finish":
                finish_error = self._finish_guard_error()
                if finish_error:
                    self.memory.append(
                        {"role": "user", "content": f"Observation: {finish_error}"}
                    )
                    logging.warning(finish_error)
                    logging.info(_LOG_SEPARATOR)
                    continue
                logging.info(f"AGENT DECIDED TO FINISH at step {iteration+1}")
                logging.info(_LOG_SEPARATOR)

                if self.synthesize_final_answer:
                    final_answer = self._generate_final_answer(task)
                else:
                    final_answer = agent_step.thought.strip()

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

            call_signature = self._call_signature(
                agent_step.action, agent_step.action_input
            )
            if call_signature in self._blocked_call_signatures:
                self._handle_blocked_call(agent_step)
                continue

            if self._detect_loop(agent_step.action, agent_step.action_input):
                self._handle_loop_detection(agent_step)
                continue

            result = self.execute_tool(agent_step.action, agent_step.action_input)
            if result.success:
                self._record_successful_execution(agent_step, result)
            else:
                self._record_failed_execution(agent_step, result)

        return (
            "Error: Maximum iterations reached without finding a solution. "
            "Please refine your request or try a different approach."
        )
