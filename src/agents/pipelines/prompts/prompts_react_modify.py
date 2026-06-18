"""Prompt templates and few-shot examples for the ReActAgentModify pipeline.

Extracted from react_modify.py to keep the agent logic readable.
Imported back via `from src.agents.pipelines.prompts.prompts_react_modify import ...`.
"""

REACT_SYSTEM_PROMPT = """You are an autonomous AI agent using the ReAct (Reasoning + Acting) framework.
        
        ### CRITICAL RULES
        1. You may ONLY use tools listed under "### AVAILABLE TOOLS" below.
        2. Tool names are CASE-SENSITIVE and must match EXACTLY.
        3. NEVER invent, guess, or modify tool names.
        4. If no tool fits → explain in Thought, then Action: Finish.
        
        Available tool names only: {tool_names}

        ### STRICT OUTPUT FORMAT
        Answer the following questions by reasoning step-by-step. Use the following format:

        Thought: <your reasoning about what to do next>
        Action: <AVAILABLE tool name> or <Finish>
        Action Input: <arguments>  # if Action is Finish then write None
        
        If Action is Finish, then write in Action Input: None
        
        When you have the final answer, use: Action: Finish
        
        ### EXAMPLES
        
        Question: What is 2 + 2 * 3?
        Thought: I need to evaluate this mathematical expression.
        Action: calculator
        Action Input: 2 + 2 * 3
        Observation: 8
        Thought: The calculation is complete.
        Action: Finish
        Action Input: None

        ### AVAILABLE TOOLS
        {tools_formatted}

        ### RULES
        IMPORTANT: Few-shot examples may demonstrate different tools for format/structure reference only. You must strictly call ONLY the available tools listed in {tool_names}.
        1. Always start with "Thought:".
        2. Use only exact tool names from the list above, or "Finish" when done.
        3. Arguments go on a separate line after Action: Action Input: <arguments>.
        4. When you have the final answer, output: Action: Finish
        5. One action per response. Stop after outputting Action.
    """

REACT_FINISH_PROMPT = """Based on the information provided, generate only the final answer for the task without Thought and Action, only final text: {task}"""

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
Action Input 1: "Write a Python function named `g` that accepts a pandas DataFrame `df` and a list of integers `List`. The function should return a new DataFrame where the rows are reordered according to the indices in `List` using the `.iloc[]` accessor."

Observation 1: Here is the code to solve the problem:

```python
def g(df, List):
    return df.iloc[List]
```

Thought 2:
The LLM suggests df.iloc[List]. To verify the exact behavior with integer lists and ensure there are no index alignment side effects, I will search the code database for concrete examples.

Action 2: db_search
Action Input 2: 'pandas iloc reorder rows by list of integers example'

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
Action Input 3: None

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
Action Input 1: "Write Python code using PyTorch to fix the following issue: 'data' has shape (10, 6, 32) and 'W' has shape (32,). Calculate the dot product such that the result has shape (10, 6). Use broadcasting (unsqueeze/expand) and sum, rather than bmm, to handle the dimensions correctly."


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
Action Input 2: 'pytorch batch dot product expand sum unsqueeze example'

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

Action 3: Finish
Action Input 3: None


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
Action Input 1: "Write a Python function named `f` that takes a NumPy array `arr` and a target tuple `shape`. It should return a new array padded with zeros to match the target shape, padding only at the bottom and right sides. Use `np.pad`."


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
Action Input 2: 'numpy pad array to target shape zeros bottom right example'

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

Action 3: Finish
Action Input 3: None

=== TASK ===
"""
