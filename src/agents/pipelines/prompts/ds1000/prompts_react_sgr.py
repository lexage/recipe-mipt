"""Prompt templates and few-shot examples for ReActAgentSGR.

Moved out of react_sgr.py to keep the agent logic readable.
Imported back via `from src.agents.pipelines.prompts.ds1000.prompts_react_sgr import ...`.
Includes archived/experimental variants (kept commented, with metric notes).
"""

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

# FEW_SHOT_AUTO_COT = """Problem: Problem:\nI want to convert a 1-dimensional array into a 2-dimensional array by specifying the number of columns in the 2D array. Something that would work like this:\n> import numpy as np\n> A = np.array([1,2,3,4,5,6,7])\n> B = vec2matrix(A,ncol=2)\n> B\narray([[1, 2],\n       [3, 4],\n       [5, 6]])\nNote that when A cannot be reshaped into a 2D array, we tend to discard elements which are at the end of A.\nDoes numpy have a function that works like my made-up function \"vec2matrix\"? (I understand that you can index a 1D array like a 2D array, but that isn't an option in the code I have - I need to make this conversion.)\nA:\n<code>\nimport numpy as np\nA = np.array([1,2,3,4,5,6,7])\nncol = 2\n</code>\nB = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n\nReasoning: === FEW-SHOT EXAMPLE 1 ===\nTASK:\nConvert a 1-dimensional array into a 2-dimensional array by specifying the number of columns in the 2D array. When the array cannot be reshaped into a 2D array, discard the excess elements at the end of the array.\n\nProblem:\nI want to convert a 1-dimensional array into a 2-dimensional array by specifying the number of columns in the 2D array. Something that would work like this:\n> import numpy as np\n> A = np.array([1,2,3,4,5,6,7])\n> B = vec2matrix(A,ncol=2)\n> B\narray([[1, 2],\n       [3, 4],\n       [5, 6]])\nNote that when A cannot be reshaped into a 2D array, we tend to discard elements which are at the end of A.\nDoes numpy have a function that works like my made-up function \"vec2matrix\"? (I understand that you can index a 1D array like a 2D array, but that isn't an option in the code I have - I need to make this conversion.)\n\nA:\n<code>\nimport numpy as np\nA = np.array([1,2,3,4,5,6,7])\nncol = 2\n</code>\nB = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\nThought 1: To solve this problem, I need to first calculate how many elements should remain after discarding the excess ones to fit the specified number of columns. Then, reshape the array accordingly.\nAction 1: llm\nAction Input 1: {'query': 'How can I convert a 1D array to a 2D array by specifying the number of columns, discarding excess elements if necessary?'}\nIs_final 1: False\n\nObservation 1: \nTo achieve the desired functionality, you can use the following approach:\n1. Calculate the total number of elements that can be included in the 2D array based on the number of columns.\n2. Slice the original array to remove excess elements.\n3. Reshape the sliced array into the desired 2D shape.\n\nThought 2: Based on the information received, I will implement the logic to slice the array and then reshape it.\nAction 2: Finish\nAction Input 2: {}\n\nCRITICAL RULES FOR INTERNAL FORMATTING:\n1. Action Input MUST be a Python dictionary using single quotes for strings: {'query': 'your query here'}.\n2. Simulated observations for 'db_search' must look like actual search chunks, for example:\n   \"query: your query\n   retrieved context: [CHUNK 12345 | doc=6789]\n   In [1]: ...\n   Out[1]: ...\"\n3. Maintain clean code formatting inside markdown blocks where appropriate.\n4. Generate the COMPLETE trajectory block from \"=== FEW-SHOT EXAMPLE...\" to the last action in one response. Do not truncate.\n\nB = (A.shape[0] // ncol) * ncol  # Calculate the number of elements to keep\nB = A[:B].reshape(-1, ncol)  # Slice the array and reshape it\n</code>\nAnswer: col = ( A.shape[0] // ncol) * ncol\nB = A[:col]\nB= np.reshape(B, (-1, ncol))\n\n\nProblem: import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nx = np.arange(10)\ny = np.arange(10)\n\n# Plot y over x and label y axis \"Y\"\n# Show y axis ticks on the left and y axis label on the right\n# SOLUTION START\n\nReasoning: === FEW-SHOT EXAMPLE 1 ===\nTASK:\nProblem:\n```python\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nx = np.arange(10)\ny = np.arange(10)\n\n# Plot y over x and label y axis \"Y\"\n# Show y axis ticks on the left and y axis label on the right\n```\n# SOLUTION START\n\nA:\n<code>\nplt.plot(x, y)\nplt.ylabel(\"Y\")\nax = plt.gca()\nax.yaxis.set_label_position(\"right\")\n</code>\nresult = plt.show() # put solution in this variable\nBEGIN SOLUTION\n<code>\nThought 1: The task requires plotting y over x and labeling the y-axis as \"Y\". Additionally, we need to move the y-axis label to the right side of the plot while keeping the y-axis ticks on the left.\nAction 1: llm\nAction Input 1: {'query': 'Plot y over x and label y axis \"Y\". Show y axis ticks on the left and y axis label on the right.'}\nIs_final 1: False\n\nObservation 1: \n```python\nplt.plot(x, y)  # Plotting y over x\nplt.ylabel(\"Y\")  # Labeling the y-axis as \"Y\"\nax = plt.gca()  # Getting the current axes instance\nax.yaxis.set_label_position(\"right\")  # Moving the y-axis label to the right\n```\n\nThought 2: Now that we have the code snippet for plotting and configuring the y-axis, we can finalize the solution by executing the plot command.\nAction 2: Finish\nAction Input 2: {}\nIs_final 2: True\n</code>\nAnswer: plt.plot(x, y)\nplt.ylabel(\"y\")\nax = plt.gca()\nax.yaxis.set_label_position(\"right\")\n\nProblem: Problem:\n\nI would like to break down a pandas column, which is the last column, consisting of a list of elements into as many columns as there are unique elements i.e. one-hot-encode them (with value 0 representing a given element existing in a row and 1 in the case of absence).\n\nFor example, taking dataframe df\n\nCol1   Col2         Col3\n C      33     [Apple, Orange, Banana]\n A      2.5    [Apple, Grape]\n B      42     [Banana]\nI would like to convert this to:\n\ndf\n\nCol1   Col2   Apple   Orange   Banana   Grape\n C      33     0        0        0       1\n A      2.5    0        1        1       0\n B      42     1        1        0       1\nSimilarly, if the original df has four columns, then should do the operation to the 4th one.\nCould any one give me any suggestion of pandas or sklearn methods? thanks!\n\nA:\n\n<code>\nimport pandas as pd\nimport numpy as np\nimport sklearn\ndf = load_data()\n</code>\ndf_out = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n\nReasoning: === FEW-SHOT EXAMPLE 1 ===\nTASK:\nProblem: I would like to break down a pandas column, which is the last column, consisting of a list of elements into as many columns as there are unique elements i.e., one-hot-encode them (with value 0 representing a given element existing in a row and 1 in the case of absence).\n\nFor example, taking dataframe `df`:\n\n```\nCol1   Col2         Col3\n C      33     [Apple, Orange, Banana]\n A      2.5    [Apple, Grape]\n B      42     [Banana]\n```\n\nI would like to convert this to:\n\n```\nCol1   Col2   Apple   Orange   Banana   Grape\n C      33     0        0        0       1\n A      2.5    0        1        1       0\n B      42     1        1        0       1\n```\n\nSimilarly, if the original `df` has four columns, then should do the operation to the 4th one.\nCould anyone give me any suggestion of pandas or sklearn methods? Thanks!\n\nA:\n\n<code>\nimport pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import MultiLabelBinarizer\ndf = load_data()\n</code>\ndf_out = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\nThought 1: The first step involves transforming the last column into a binary matrix where each column represents a unique item from the list within the cells of the last column.\nAction 1: llm\nAction Input 1: {'query': 'transform last column of a pandas dataframe into a binary matrix for each unique item'}\nIs_final 1: False\n\nObservation 1: \nTo achieve the desired transformation, we can use the `MultiLabelBinarizer` from `sklearn.preprocessing`. This will allow us to encode the lists in the last column into a binary format. We then need to invert the binary values (0 becomes 1 and vice versa) to match the requirement.\n\nThought 2: We need to invert the binary matrix after encoding the lists in the last column.\nAction 2: llm\nAction Input 2: {'query': 'invert binary values in a pandas dataframe'}\nIs_final 2: False\n\nObservation 2: \nTo invert the binary values in a DataFrame, we can subtract the DataFrame from 1 (`1 - df`). This operation will change all 0s to 1s and all 1s to 0s.\n\nThought 3: Combine both actions to achieve the final solution.\nAction 3: Finish\nAction Input 3: {}\n\nCRITICAL RULES FOR INTERNAL FORMATTING:\n1. Action Input MUST be a Python dictionary using single quotes for strings: {'query': 'your query here'}. Do NOT use pure JSON format.\n2. Simulated observations for 'db_search' must look like actual search chunks, for example:\n   \"query: your query\n   retrieved context: [CHUNK 12345 | doc=6789]\n   In [1]: ...\n   Out[1]: ...\"\n3. Maintain clean code formatting inside markdown blocks where appropriate.\n4. Generate the COMPLETE trajectory block from \"=== FEW-SHOT EXAMPLE...\" to the last action in one response. Do not truncate.\n\n```python\nmlb = MultiLabelBinarizer()\n# Transform the last column into a binary matrix\nbinary_matrix = pd.DataFrame(mlb.fit_transform(df.pop(df.columns[-1])), index=df.index, columns=mlb.classes_)\n# Invert the binary matrix\ninverted_matrix = 1 - binary_matrix\n# Join the inverted matrix back to the original dataframe\ndf_out = df.join(inverted_matrix)\n```\nAnswer: from sklearn.preprocessing import MultiLabelBinarizer\n\nmlb = MultiLabelBinarizer()\n\ndf_out = df.join(\n    pd.DataFrame(\n        mlb.fit_transform(df.pop(df.columns[-1])),\n        index=df.index,\n        columns=mlb.classes_))\nfor idx in df_out.index:\n    for col in mlb.classes_:\n        df_out.loc[idx, col] = 1 - df_out.loc[idx, col]\n\n"""

FEW_SHOT_AUTO_COT = """=== FEW-SHOT EXAMPLE 1===\nTASK:\nProblem: I want to convert a 1-dimensional array into a 2-dimensional array by specifying the number of columns in the 2D array. Something that would work like this:\n> import numpy as np\n> A = np.array([1,2,3,4,5,6,7])\n> B = vec2matrix(A,ncol=2)\n> B\narray([[1, 2],\n       [3, 4],\n       [5, 6]])\nNote that when A cannot be reshaped into a 2D array, we tend to discard elements which are at the end of A.\nDoes numpy have a function that works like my made-up function \"vec2matrix\"? (I understand that you can index a 1D array like a 2D array, but that isn't an option in the code I have - I need to make this conversion.)\n\nA:\n<code>\nimport numpy as np\nA = np.array([1,2,3,4,5,6,7])\nncol = 2\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\nThought 1: The numpy library does not have a direct function named \"vec2matrix\". However, we can achieve the desired functionality by first trimming the array to a size that fits the specified number of columns and then reshaping it.\nAction 1: llm\nAction Input 1: {'query': 'How do I reshape a 1D numpy array into a 2D array while discarding extra elements if necessary?'}\nIs_final 1: False\n\nObservation 1: To reshape a 1D array into a 2D array with a specific number of columns, you can first calculate the largest multiple of the column count that is less than or equal to the length of the array. Then, slice the array up to this point and reshape it using `numpy.reshape`.\n\nThought 2: Let's implement the logic based on the observation to trim the array and reshape it.\nAction 2: llm\nAction Input 2: {'query': 'Implement the logic to reshape a 1D numpy array into a 2D array with a given number of columns, discarding extra elements if necessary.'}\nIs_final 2: False\n\nObservation 2: Here is the implementation based on the previous discussion:\n```python\ncol = (A.shape[0] // ncol) * ncol\ntrimmed_A = A[:col]\nresult = trimmed_A.reshape(-1, ncol)\n```\n\nThought 3: The implementation has been completed and the solution is ready.\nAction 3: Finish\nAction Input 3: {}\nIs_final 3: True\n</code>\nAnswer: col = ( A.shape[0] // ncol) * ncol\nB = A[:col]\nB= np.reshape(B, (-1, ncol))\n\n\n=== FEW-SHOT EXAMPLE 2===\nTASK:\nProblem: import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nx = np.arange(10)\ny = np.arange(10)\n\n# Plot y over x and label y axis \"Y\"\n# Show y axis ticks on the left and y axis label on the right\n# SOLUTION START\n\nA:\n<code>\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nx = np.arange(10)\ny = np.arange(10)\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\nThought 1: To plot y over x and set up the y-axis label on the right side, we need to adjust the current plotting configuration.\nAction 1: llm\nAction Input 1: {'query': 'Plot y over x, label y axis \"Y\", show y axis ticks on the left and y axis label on the right'}\nIs_final 1: False\n\nObservation 1: The task requires modifying the current plotting setup to ensure the y-axis label appears on the right while keeping the y-axis ticks on the left. This can be achieved by adjusting the axis properties.\n\nThought 2: Proceeding with the plotting and customizing the y-axis settings.\nAction 2: llm\nAction Input 2: {'query': 'Adjust y-axis settings to show ticks on the left and label on the right'}\nIs_final 2: False\n\nObservation 2: Adjusting the y-axis to display the label on the right side while keeping the ticks on the left involves setting the label position property of the y-axis to 'right'.\n\nThought 3: Finalize the solution by implementing the plotting and customization.\nAction 3: Finish\nAction Input 3: {}\nIs_final 3: True\n</code>\nAnswer: plt.plot(x, y)\nplt.ylabel(\"y\")\nax = plt.gca()\nax.yaxis.set_label_position(\"right\")\n\n=== FEW-SHOT EXAMPLE 3===\nTASK:\nProblem: I would like to break down a pandas column, which is the last column, consisting of a list of elements into as many columns as there are unique elements i.e. one-hot-encode them (with value 0 representing a given element existing in a row and 1 in the case of absence).\n\nA:\n<code>\nfrom sklearn.preprocessing import MultiLabelBinarizer\nimport pandas as pd\n</code>\ndf_out = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\nThought 1: The problem requires transforming the last column of a DataFrame containing lists of items into multiple columns, effectively one-hot encoding those items. The values should be flipped so that presence is marked with 0 and absence with 1.\nAction 1: llm\nAction Input 1: {'query': 'Transform the last column of a DataFrame into multiple one-hot encoded columns, but flip the binary values so that presence is marked with 0 and absence with 1.'}\nIs_final 1: False\n\nObservation 1: \nTo achieve the transformation described, we can use `sklearn.preprocessing.MultiLabelBinarizer` to first one-hot encode the lists in the last column. Then, we can join these new columns back to the original DataFrame. Finally, we will invert the binary values to match the desired output format.\n\nThought 2: Implement the described logic using MultiLabelBinarizer and DataFrame operations.\nAction 2: llm\nAction Input 2: {'query': 'Implement the logic using MultiLabelBinarizer and DataFrame operations.'}\n\nObservation 2: \n```python\nmlb = MultiLabelBinarizer()\n# Transform the last column into a one-hot encoded format\none_hot_encoded = mlb.fit_transform(df[df.columns[-1]])\n# Create a DataFrame from the transformed data\nencoded_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)\n# Join the new DataFrame with the original, excluding the last column\ndf_out = df.iloc[:, :-1].join(encoded_df)\n# Flip the binary values so that presence is 0 and absence is 1\ndf_out[df.columns[-1]] = df[df.columns[-1]]\ndf_out[mlb.classes_] = 1 - df_out[mlb.classes_]\n```\n\nThought 3: The solution is ready, with the DataFrame `df_out` containing the transformed columns as required.\nAction 3: Finish\nAction Input 3: {}\nIs_final 3: True\n</code>\nAnswer: from sklearn.preprocessing import MultiLabelBinarizer\n\nmlb = MultiLabelBinarizer()\n\ndf_out = df.join(\n    pd.DataFrame(\n        mlb.fit_transform(df.pop(df.columns[-1])),\n        index=df.index,\n        columns=mlb.classes_))\nfor idx in df_out.index:\n    for col in mlb.classes_:\n        df_out.loc[idx, col] = 1 - df_out.loc[idx, col]\n\n"""


FEW_SHOT_REGISTRY = {
    "zero_shot": "",
    "cot": FEW_SHOT_COT_EXAMPLES,
    "contrastive_cot": FEW_SHOT_CONTRASTIVE_COT,
    "auto_cot": FEW_SHOT_AUTO_COT
}
