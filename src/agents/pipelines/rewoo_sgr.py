from typing import List, Union, Dict, Any, Type
import logging
import json
import re

from pydantic import BaseModel, create_model, Field
from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.tools import BaseTool, LLMTool


PLANNER_PROMPT = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools.

TOOLS (Case-Sensitive, Exact Names Only)
{tools_formatted}

You must respond with a structured JSON format containing a list of steps. Each step should have:
- step_id: integer (starting from 1)
- plan: string description of what this step does
- tool: string name of the tool to use
- args: object with tool-specific arguments
- evidence_tag: string like "#E1", "#E2", etc.
- depends_on: list of evidence tags this step depends on (e.g., ["#E1"] or [])

CONSTRAINTS
1. Action MUST be one of: {tool_names} or 'finish'.
   IMPORTANT: Few-shot examples may demonstrate different tools for format/structure reference only. You must strictly call ONLY the available tools listed in {tool_names}.

For example:
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?

{{
  "steps": [
    {{
      "step_id": 1,
      "plan": "Given Thomas worked x hours, translate the problem into algebraic expressions and solve with Wolfram Alpha",
      "tool": "WolframAlpha",
      "args": {{"query": "Solve x + (2x − 10) + ((2x − 10) − 8) = 157"}},
      "evidence_tag": "#E1",
      "depends_on": []
    }},
    {{
      "step_id": 2,
      "plan": "Find out the number of hours Thomas worked",
      "tool": "llm",
      "args": {{"query": "What is x, given #E1"}},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }},
    {{
      "step_id": 3,
      "plan": "Calculate the number of hours Rebecca worked",
      "tool": "calculator",
      "args": {{"expression": "(2 * #E2 - 10) - 8"}},
      "evidence_tag": "#E3",
      "depends_on": ["#E2"]
    }}
  ]
}}

Begin!
Describe your plans with rich details.

Task: {task}"""

REWOO_FEW_SHOT_EXAMPLES = """
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

{
  "steps": [
    {
      "step_id": 1,
      "plan": "Generate a Python function using pandas .iloc[] to reorder the DataFrame rows based on the provided list of positional indices.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `g` that accepts a pandas DataFrame `df` and a list of integers `List`. The function should return a new DataFrame where the rows are reordered according to the indices in `List` using the `.iloc[]` accessor."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Verify the behavior of pandas .iloc[] with a list of integers in the documentation database to ensure no alignment side-effects occur.",
      "tool": "db_search",
      "args": {"query": "pandas iloc reorder rows by list of integers example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}

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

{
  "steps": [
    {
      "step_id": 1,
      "plan": "Ask the LLM to generate code using PyTorch broadcasting (unsqueeze and expand) and summation to calculate the batch dot product between `data` of shape (10, 6, 32) and `W` of shape (32,).",
      "tool": "llm",
      "args": {"query": "Write Python code using PyTorch to fix the following issue: 'data' has shape (10, 6, 32) and 'W' has shape (32,). Calculate the dot product such that the result has shape (10, 6). Use broadcasting (unsqueeze/expand) and sum, rather than bmm, to handle the dimensions correctly."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the code database to verify standard PyTorch implementations for batch dot products utilizing expand and sum to ensure optimization and correctness.",
      "tool": "db_search",
      "args": {"query": "pytorch batch dot product expand sum unsqueeze example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}

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

{
  "steps": [
    {
      "step_id": 1,
      "plan": "Generate a Python function using `np.pad` to add zero padding dynamically to the bottom and right of a 2D array until it matches a target shape.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `f` that takes a NumPy array `arr` and a target tuple `shape`. It should return a new array padded with zeros to match the target shape, padding only at the bottom and right sides. Use `np.pad`."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the documentation database for zero padding examples using dynamic pad width calculations in NumPy to validate the padding schema.",
      "tool": "db_search",
      "args": {"query": "numpy pad array to target shape zeros bottom right example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}
"""

REWOO_FEW_SHOT_COT_EXAMPLES = """
=== FEW-SHOT EXAMPLE 1 ===
TASK:
Problem:
I have the following DataFrame:
    Col1  Col2  Col3  Type
0      1     2     3     1
1      4     5     6     1
2      7     8     9     2
3     10    11    12     2
4     13    14    15     3
5     16    17    18     3

The DataFrame is read from a CSV file. All rows which have Type 1 are on top, followed by the rows with Type 2, followed by the rows with Type 3, etc.
I would like to shuffle the order of the DataFrame's rows according to a list.
For example, give a list [2, 4, 0, 3, 1, 5] and desired result should be:
    Col1  Col2  Col3  Type
2      7     8     9     2
4     13    14    15     3
0      1     2     3     1
3     10    11    12     2
1      4     5     6     1
5     16    17    18     3
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

THOUGHT:
The objective is to reorder a pandas DataFrame based on a user-provided list of position-based indices. 
1. To select or reorder rows by their absolute integer positions in pandas, the `.iloc` accessor is the appropriate tool. 
2. I need to formulate a plan where an LLM creates a robust function utilizing `.iloc` with the list.
3. To confirm best practices and verify that passing an array/list to `.iloc` behaves predictably without re-indexing side-effects, I will also issue a documentation database search.
Thus, Step 1 will handle generation, and Step 2 will handle validation.

BEGIN SOLUTION
<code>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Generate a Python function using pandas .iloc[] to reorder the DataFrame rows based on the provided list of positional indices.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `g` that accepts a pandas DataFrame `df` and a list of integers `List`. The function should return a new DataFrame where the rows are reordered according to the indices in `List` using the `.iloc[]` accessor."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Verify the behavior of pandas .iloc[] with a list of integers in the documentation database to ensure no alignment side-effects occur.",
      "tool": "db_search",
      "args": {"query": "pandas iloc reorder rows by list of integers example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}
</code>

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

THOUGHT:
The user is facing a dimensional mismatch error in PyTorch. They have `data` of shape (10, 6, 32) and weights `W` of shape (32,), and want a dot product resulting in shape (10, 6).
1. `torch.bmm` expects two 3D tensors, so it fails here because `W` is 1D.
2. The correct approach in PyTorch is either using `torch.matmul` (which handles broadcasting) or explicitly unsqueezing `W` to align dimensions, expanding it, and performing an element-wise multiplication followed by a sum over the last dimension.
3. I will instruct the LLM to generate code using the broadcasting/sum approach to fix this error explicitly.
4. Additionally, I should query the internal database for PyTorch batch dot product examples using `expand` and `sum` to ensure the generated pattern is idiomatic and performant.

BEGIN SOLUTION
<code>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Ask the LLM to generate code using PyTorch broadcasting (unsqueeze and expand) and summation to calculate the batch dot product between `data` of shape (10, 6, 32) and `W` of shape (32,).",
      "tool": "llm",
      "args": {"query": "Write Python code using PyTorch to fix the following issue: 'data' has shape (10, 6, 32) and 'W' has shape (32,). Calculate the dot product such that the result has shape (10, 6). Use broadcasting (unsqueeze/expand) and sum, rather than bmm, to handle the dimensions correctly."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the code database to verify standard PyTorch implementations for batch dot products utilizing expand and sum to ensure optimization and correctness.",
      "tool": "db_search",
      "args": {"query": "pytorch batch dot product expand sum unsqueeze example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}
</code>

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
### TEXT THOUGHT:
The goal is to dynamically pad a 2D NumPy array with zeros so that it reaches a specified target shape (93, 13), specifically adding padding only to the right and bottom.
1. The standard NumPy function for adding padding is `np.pad`.
2. To pad only the right and bottom, the padding width format should be `((0, bottom_pad), (0, right_pad))`, where `bottom_pad = target_rows - current_rows` and `right_pad = target_cols - current_cols`.
3. I will create a step for the LLM to generate a robust Python function implementing this math using `np.pad`.
4. I will also create a verification step to search the database for dynamic 2D padding templates to confirm syntax accuracy.

BEGIN SOLUTION
<code>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Generate a Python function using `np.pad` to add zero padding dynamically to the bottom and right of a 2D array until it matches a target shape.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `f` that takes a NumPy array `arr` and a target tuple `shape`. It should return a new array padded with zeros to match the target shape, padding only at the bottom and right sides. Use `np.pad`."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the documentation database for zero padding examples using dynamic pad width calculations in NumPy to validate the padding schema.",
      "tool": "db_search",
      "args": {"query": "numpy pad array to target shape zeros bottom right example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}
</code>
"""

REWOO_FEW_SHOT_CONTRASTIVE = """
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
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Generate a Python function using pandas .iloc[] to reorder the DataFrame rows based on the provided list of positional indices.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `g` that accepts a pandas DataFrame `df` and a list of integers `List`. The function should return a new DataFrame where the rows are reordered according to the indices in `List` using the `.iloc[]` accessor."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Verify the behavior of pandas .iloc[] with a list of integers in the documentation database to ensure no alignment side-effects occur.",
      "tool": "db_search",
      "args": {"query": "pandas iloc reorder rows by list of integers example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}

[WRONG TRAJECTORY]
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Generate a Python function using pandas .iloc[] to reorder the DataFrame rows based on the provided list of positional indices.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `g` that accepts a pandas DataFrame `df` and a list of integers `List`. The function should return a new DataFrame where the rows are reordered according to the indices in `List` using the `.iloc[]` accessor."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Execute the code to verify it runs without syntax or runtime errors.",
      "tool": "code_executor",
      "args": {"code": "import pandas as pd; df = pd.DataFrame({'A': [1,2,3]}); print(df.iloc)"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    },
    {
      "step_id": 3,
      "plan": "Search the database for the exact same query multiple times to be extra sure.",
      "tool": "db_search",
      "args": {"query": "pandas iloc reorder rows by list of integers example"},
      "evidence_tag": "#E3",
      "depends_on": []
    }
  ]
}
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
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Ask the LLM to generate code using PyTorch broadcasting (unsqueeze and expand) and summation to calculate the batch dot product between `data` of shape (10, 6, 32) and `W` of shape (32,).",
      "tool": "llm",
      "args": {"query": "Write Python code using PyTorch to fix the following issue: 'data' has shape (10, 6, 32) and 'W' has shape (32,). Calculate the dot product such that the result has shape (10, 6). Use broadcasting (unsqueeze/expand) and sum, rather than bmm, to handle the dimensions correctly."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the code database to verify standard PyTorch implementations for batch dot products utilizing expand and sum to ensure optimization and correctness.",
      "tool": "db_search",
      "args": {"query": "pytorch batch dot product expand sum unsqueeze example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}

[WRONG TRAJECTORY]
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Ask the LLM to generate code using PyTorch broadcasting.",
      "tool": "llm",
      "args": {"query": "Write Python code using PyTorch to fix the following issue: 'data' has shape (10, 6, 32) and 'W' has shape (32,). Calculate the dot product such that the result has shape (10, 6)."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the database for the example pattern using a raw string argument instead of a JSON object mapping.",
      "tool": "db_search",
      "args": "pytorch batch dot product expand sum unsqueeze example",
      "evidence_tag": "#E2",
      "depends_on": []
    },
    {
      "step_id": 3,
      "plan": "Re-run the exact same search step to check for data updates.",
      "tool": "db_search",
      "args": {"query": "pytorch batch dot product expand sum unsqueeze example"},
      "evidence_tag": "#E3",
      "depends_on": []
    }
  ]
}
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
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Generate a Python function using `np.pad` to add zero padding dynamically to the bottom and right of a 2D array until it matches a target shape.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `f` that takes a NumPy array `arr` and a target tuple `shape`. It should return a new array padded with zeros to match the target shape, padding only at the bottom and right sides. Use `np.pad`."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the documentation database for zero padding examples using dynamic pad width calculations in NumPy to validate the padding schema.",
      "tool": "db_search",
      "args": {"query": "numpy pad array to target shape zeros bottom right example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}

[WRONG TRAJECTORY]
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Request the function code from LLM.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `f` that takes a NumPy array `arr` and a target tuple `shape` to zero pad it."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Finish the task execution immediately without checking code library specifications.",
      "tool": "finish",
      "args": {},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}
"""

REWOO_FEW_SHOT_CONTRASTIVE_COT = """
=== FEW-SHOT EXAMPLE 1 ===
TASK:
Problem:
I have the following DataFrame:
    Col1  Col2  Col3  Type
0      1     2     3     1
1      4     5     6     1
2      7     8     9     2
3     10    11    12     2
4     13    14    15     3
5     16    17    18     3

The DataFrame is read from a CSV file. All rows which have Type 1 are on top, followed by the rows with Type 2, followed by the rows with Type 3, etc.
I would like to shuffle the order of the DataFrame's rows according to a list.
For example, give a list [2, 4, 0, 3, 1, 5] and desired result should be:
    Col1  Col2  Col3  Type
2      7     8     9     2
4     13    14    15     3
0      1     2     3     1
3     10    11    12     2
1      4     5     6     1
5     16    17    18     3
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

[CORRECT TRAJECTORY]
THOUGHT:
The user wants to reorder a pandas DataFrame based on a list of absolute integer positions. 
1. In pandas, integer-location based indexing is strictly handled by `.iloc[]`. 
2. The optimal plan is to have the LLM generate a function using `.iloc[]`, and then verify via `db_search` that passing a standard list of integers to `.iloc[]` behaves predictably without formatting bugs. This creates a clean, minimal, and fully verified 2-step plan.

<code>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Generate a Python function using pandas .iloc[] to reorder the DataFrame rows based on the provided list of positional indices.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `g` that accepts a pandas DataFrame `df` and a list of integers `List`. The function should return a new DataFrame where the rows are reordered according to the indices in `List` using the `.iloc[]` accessor."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Verify the behavior of pandas .iloc[] with a list of integers in the documentation database to ensure no alignment side-effects occur.",
      "tool": "db_search",
      "args": {"query": "pandas iloc reorder rows by list of integers example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}
</code>

[WRONG TRAJECTORY]
THOUGHT:
This trajectory represents a flawed thinking process. 
1. It attempts to use `code_executor` prematurely with an incomplete and hardcoded script snippet that only references `.iloc` without actually invoking or testing the requested function logic. 
2. It introduces loop redundancy by creating Step 3 to search the exact same query multiple times in `db_search`, which wastes tokens and API calls without gathering new evidence or optimization insights.

<code>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Generate a Python function using pandas .iloc[] to reorder the DataFrame rows based on the provided list of positional indices.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `g` that accepts a pandas DataFrame `df` and a list of integers `List`. The function should return a new DataFrame where the rows are reordered according to the indices in `List` using the `.iloc[]` accessor."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Execute the code to verify it runs without syntax or runtime errors.",
      "tool": "code_executor",
      "args": {"code": "import pandas as pd; df = pd.DataFrame({'A': [1,2,3]}); print(df.iloc)"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    },
    {
      "step_id": 3,
      "plan": "Search the database for the exact same query multiple times to be extra sure.",
      "tool": "db_search",
      "args": {"query": "pandas iloc reorder rows by list of integers example"},
      "evidence_tag": "#E3",
      "depends_on": []
    }
  ]
}
</code>

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

[CORRECT TRAJECTORY]
THOUGHT:
The goal is to compute a dot product between a batched 3D tensor and a 1D tensor. 
1. `torch.bmm` requires both inputs to be 3D, hence the crash. The proper way is to use PyTorch broadcasting and dimension expansion (`unsqueeze` + `expand` or `matmul`), followed by summing along the correct axis.
2. Step 1 will ask the LLM for code that specifically leverages explicit broadcasting/summing to avoid ambiguity.
3. Step 2 will query the library database using valid JSON arguments to verify the syntax for batch dot products.

<code>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Ask the LLM to generate code using PyTorch broadcasting (unsqueeze and expand) and summation to calculate the batch dot product between `data` of shape (10, 6, 32) and `W` of shape (32,).",
      "tool": "llm",
      "args": {"query": "Write Python code using PyTorch to fix the following issue: 'data' has shape (10, 6, 32) and 'W' has shape (32,). Calculate the dot product such that the result has shape (10, 6). Use broadcasting (unsqueeze/expand) and sum, rather than bmm, to handle the dimensions correctly."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the code database to verify standard PyTorch implementations for batch dot products utilizing expand and sum to ensure optimization and correctness.",
      "tool": "db_search",
      "args": {"query": "pytorch batch dot product expand sum unsqueeze example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}
</code>

[WRONG TRAJECTORY]
THOUGHT:
This trajectory shows multiple execution faults.
1. The LLM query in Step 1 is too generic and fails to specify the required constraint (broadcasting/sum vs bmm), which often leads to inaccurate code generation.
2. In Step 2, the `args` parameter is passed incorrectly as a raw text string instead of the required structured JSON map `{"query": "..."}`. This will cause a tool schema parsing error.
3. Step 3 introduces a duplicate, redundant database search identical to Step 2 without analyzing any new state, creating a loop.

<code>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Ask the LLM to generate code using PyTorch broadcasting.",
      "tool": "llm",
      "args": {"query": "Write Python code using PyTorch to fix the following issue: 'data' has shape (10, 6, 32) and 'W' has shape (32,). Calculate the dot product such that the result has shape (10, 6)."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the database for the example pattern using a raw string argument instead of a JSON object mapping.",
      "tool": "db_search",
      "args": "pytorch batch dot product expand sum unsqueeze example",
      "evidence_tag": "#E2",
      "depends_on": []
    },
    {
      "step_id": 3,
      "plan": "Re-run the exact same search step to check for data updates.",
      "tool": "db_search",
      "args": {"query": "pytorch batch dot product expand sum unsqueeze example"},
      "evidence_tag": "#E3",
      "depends_on": []
    }
  ]
}
</code>

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
THOUGHT:
The user needs to pad a 2D array dynamically up to a target size of (93, 13) using zeros, but only extending the bottom and right margins.
1. The standard way to achieve asymmetric padding in NumPy is `np.pad(arr, ((0, pad_row), (0, pad_col)), 'constant')`.
2. Step 1 plans an LLM prompt that explicitly mentions the target sides (bottom and right) and the `np.pad` requirement.
3. Step 2 validates the dynamic math setup against the code database to prevent off-by-one errors during deployment on thousands of rows.

<code>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Generate a Python function using `np.pad` to add zero padding dynamically to the bottom and right of a 2D array until it matches a target shape.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `f` that takes a NumPy array `arr` and a target tuple `shape`. It should return a new array padded with zeros to match the target shape, padding only at the bottom and right sides. Use `np.pad`."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the documentation database for zero padding examples using dynamic pad width calculations in NumPy to validate the padding schema.",
      "tool": "db_search",
      "args": {"query": "numpy pad array to target shape zeros bottom right example"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}
</code>

[WRONG TRAJECTORY]
THOUGHT:
This plan shows reckless execution logic.
1. In Step 1, the LLM prompt is overly broad; it lacks structural requirements (e.g., specifying asymmetric right/bottom padding), meaning the generated code will likely pad uniformly or incorrectly.
2. In Step 2, the agent attempts to complete and `finish` the task immediately without any database cross-checking, validation, or dependency tracking, risking silent runtime failures in a production dataset.

<code>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Request the function code from LLM.",
      "tool": "llm",
      "args": {"query": "Write a Python function named `f` that takes a NumPy array `arr` and a target tuple `shape` to zero pad it."},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Finish the task execution immediately without checking code library specifications.",
      "tool": "finish",
      "args": {},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}
</code>
"""

REWOO_FEW_SHOT_AUTO_COT = """=== FEW-SHOT EXAMPLE 1 ===\nTASK:\nProblem:\nI want to convert a 1-dimensional array into a 2-dimensional array by specifying the number of columns in the 2D array. Something that would work like this:\n> import numpy as np\n> A = np.array([1,2,3,4,5,6,7])\n> B = vec2matrix(A,ncol=2)\n> B\narray([[1, 2],\n       [3, 4],\n       [5, 6]])\nNote that when A cannot be reshaped into a 2D array, we tend to discard elements which are at the end of A.\nDoes numpy have a function that works like my made-up function \"vec2matrix\"? (I understand that you can index a 1D array like a 2D array, but that isn't an option in the code I have - I need to make this conversion.)\n\nA:\n<code>\nimport numpy as np\nA = np.array([1,2,3,4,5,6,7])\nncol = 2\n</code>\nB = ... # put solution in this variable\n\nTHOUGHT:\nTo solve this problem, we will first determine the maximum number of rows that can be formed with the given number of columns without exceeding the length of the array. Then, we'll slice the array to fit this new shape and reshape it accordingly. This method ensures that any excess elements are discarded.\n\nBEGIN SOLUTION\n<code>\n{\n  \"steps\": [\n    {\n      \"step_id\": 1,\n      \"plan\": \"Calculate the maximum number of elements that can be reshaped into a 2D array with the specified number of columns.\",\n      \"tool\": \"llm\",\n      \"args\": {\"query\": \"Calculate max elements for 2D array\"},\n      \"evidence_tag\": \"#E1\",\n      \"depends_on\": []\n    },\n    {\n      \"step_id\": 2,\n      \"plan\": \"Slice the original array to fit the calculated size and reshape it into a 2D array.\",\n      \"tool\": \"llm\",\n      \"args\": {\"query\": \"Slice and reshape array to 2D\"},\n      \"evidence_tag\": \"#E2\",\n      \"depends_on\": [\"#E1\"]\n    }\n  ]\n}\n</code>\nAnswer: col = ( A.shape[0] // ncol) * ncol\nB = A[:col]\nB= np.reshape(B, (-1, ncol))\n\n\n=== FEW-SHOT EXAMPLE 2 ===\nTASK:\nProblem:\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nx = np.arange(10)\ny = np.arange(10)\n\n# Plot y over x and label y axis \"Y\"\n# Show y axis ticks on the left and y axis label on the right\n# SOLUTION START\n\nA:\n<code>\nplt.plot(x, y)\nplt.ylabel(\"Y\")\nax = plt.gca()\nax.yaxis.set_label_position(\"right\")\n</code>\nresult = ax # put solution in this variable\n\nTHOUGHT:\nFirstly, we need to plot the data using matplotlib's plot function. Then, we will label the y-axis appropriately and adjust its position to be on the right side of the plot. We will use matplotlib's get current axes method to manipulate the y-axis properties.\n\nBEGIN SOLUTION\n<code>\n{\n  \"steps\": [\n    {\n      \"step_id\": 1,\n      \"plan\": \"Plot y over x using matplotlib's plot function.\",\n      \"tool\": \"llm\",\n      \"args\": {\"query\": \"Generate code to plot y over x using matplotlib.\"},\n      \"evidence_tag\": \"#E1\",\n      \"depends_on\": []\n    },\n    {\n      \"step_id\": 2,\n      \"plan\": \"Label the y-axis with 'Y'.\",\n      \"tool\": \"llm\",\n      \"args\": {\"query\": \"Generate code to set the y-axis label to 'Y' using matplotlib.\"},\n      \"evidence_tag\": \"#E2\",\n      \"depends_on\": [\"#E1\"]\n    },\n    {\n      \"step_id\": 3,\n      \"plan\": \"Set the y-axis label position to the right.\",\n      \"tool\": \"llm\",\n      \"args\": {\"query\": \"Generate code to move the y-axis label to the right side of the plot using matplotlib.\"},\n      \"evidence_tag\": \"#E3\",\n      \"depends_on\": [\"#E2\"]\n    }\n  ]\n}\n</code>\nAnswer: plt.plot(x, y)\nplt.ylabel(\"y\")\nax = plt.gca()\nax.yaxis.set_label_position(\"right\")\n\n=== FEW-SHOT EXAMPLE 3 ===\nTASK:\nProblem:\nI would like to break down a pandas column, which is the last column, consisting of a list of elements into as many columns as there are unique elements i.e. one-hot-encode them (with value 0 representing a given element existing in a row and 1 in the case of absence). For example, taking dataframe df:\n\nCol1   Col2         Col3\n C      33     [Apple, Orange, Banana]\n A      2.5    [Apple, Grape]\n B      42     [Banana]\n\nI would like to convert this to:\n\ndf\n\nCol1   Col2   Apple   Orange   Banana   Grape\n C      33     0        0        0       1\n A      2.5    0        1        1       0\n B      42     1        1        0       1\n\nSimilarly, if the original df has four columns, then should do the operation to the 4th one. Could any one give me any suggestion of pandas or sklearn methods? thanks!\n\nA:\n<code>\nimport pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import MultiLabelBinarizer\ndf = load_data()\n</code>\ndf_out = ... # put solution in this variable\n\nTHOUGHT:\nTo achieve the desired transformation, we will use the MultiLabelBinarizer from sklearn to one-hot encode the list in the last column. We will then invert the binary values to match the requirement of 0 for presence and 1 for absence. This approach is chosen because MultiLabelBinarizer efficiently handles the conversion of lists into one-hot encoded format.\n\nBEGIN SOLUTION\n<code>\n{\n  \"steps\": [\n    {\n      \"step_id\": 1,\n      \"plan\": \"Initialize the MultiLabelBinarizer to prepare for encoding the lists in the last column.\",\n      \"tool\": \"llm\",\n      \"args\": {\"query\": \"from sklearn.preprocessing import MultiLabelBinarizer; mlb = MultiLabelBinarizer()\"},\n      \"evidence_tag\": \"#E1\",\n      \"depends_on\": []\n    },\n    {\n      \"step_id\": 2,\n      \"plan\": \"Transform the last column of the DataFrame using the MultiLabelBinarizer and join the result back to the original DataFrame.\",\n      \"tool\": \"llm\",\n      \"args\": {\"query\": \"df_out = df.join(pd.DataFrame(mlb.fit_transform(df.pop(df.columns[-1])), index=df.index, columns=mlb.classes_))\"},\n      \"evidence_tag\": \"#E2\",\n      \"depends_on\": [\"#E1\"]\n    },\n    {\n      \"step_id\": 3,\n      \"plan\": \"Invert the binary values in the new columns so that 0 represents the presence of an item and 1 represents its absence.\",\n      \"tool\": \"llm\",\n      \"args\": {\"query\": \"for idx in df_out.index: for col in mlb.classes_: df_out.loc[idx, col] = 1 - df_out.loc[idx, col]\"},\n      \"evidence_tag\": \"#E3\",\n      \"depends_on\": [\"#E2\"]\n    }\n  ]\n}\n</code>\nAnswer: from sklearn.preprocessing import MultiLabelBinarizer\n\nmlb = MultiLabelBinarizer()\n\ndf_out = df.join(\n    pd.DataFrame(\n        mlb.fit_transform(df.pop(df.columns[-1])),\n        index=df.index,\n        columns=mlb.classes_))\nfor idx in df_out.index:\n    for col in mlb.classes_:\n        df_out.loc[idx, col] = 1 - df_out.loc[idx, col]\n\n"""

WORKER_PROMPT = """Respond in short directly with no extra words.\n\n{request}"""


SOLVER_PROMPT = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""

SOLVER_REWOO_FEW_SHOT_COT_EXAMPLES = """
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

- Plan: 'Generate a Python function using pandas .iloc[] to reorder the DataFrame rows based on the provided list of positional indices.'
- Evidence: 'def g(df, List):\n    return df.iloc[List]'
- Plan: 'Verify the behavior of pandas .iloc[] with a list of integers in the documentation database to ensure no alignment side-effects occur.'
- Evidence: 'The .iloc indexer in pandas purely selects rows based on integer position, making it perfect for reordering by positional index without modifying the index names directly.'

RESPONSE:
{
  "thought": "The user wants to reorder a pandas DataFrame based on a list of positional indices. Based on the evidence, `df.iloc[List]` correctly shuffles rows using positional indexing. I need to define the function `g(df, List)` and assign the result to the `result` variable as requested in the template.",
  "response": "def g(df, List):\n    return df.iloc[List]\n\nresult = g(df.copy(), List)"
}

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

- Plan: 'Ask the LLM to generate code using PyTorch broadcasting (unsqueeze and expand) and summation to calculate the batch dot product between `data` of shape (10, 6, 32) and `W` of shape (32,).'
- Evidence: 'W = W.unsqueeze(0).unsqueeze(0).expand(*data.size())\nresult = torch.sum(data * W, -1)'
- Plan: 'Search the code database to verify standard PyTorch implementations for batch dot products utilizing expand and sum to ensure optimization and correctness.'
- Evidence: 'To compute batch dot product with a 1D tensor, unsqueezing and expanding or directly utilizing broadcasting during element-wise multiplication is standard before reducing along the feature dimension using torch.sum().'

RESPONSE:
{
  "thought": "The user encounters a shape error with `torch.bmm`. The evidence provides a clean way to perform the operation by unsqueezing `W` to make it compatible with `data`'s dimensions, multiplying element-wise, summing over the last axis, and reshaping back to (10, 2, 3).",
  "response": "W = W.unsqueeze(0).unsqueeze(0).expand(*data.size())\nresult = torch.sum(data * W, 2)\nresult = result.view(10, 2, 3)"
}

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

- Plan: 'Generate a Python function using `np.pad` to add zero padding dynamically to the bottom and right of a 2D array until it matches a target shape.'
- Evidence: 'result = np.pad(arr, ((0, shape[0]-arr.shape[0]), (0, shape[1]-arr.shape[1])), 'constant')'
- Plan: 'Search the documentation database for zero padding examples using dynamic pad width calculations in NumPy to validate the padding schema.'
- Evidence: 'NumPy pad function syntax requires a tuple of pads for each dimension: ((top, bottom), (left, right)). To pad only bottom and right, top and left should be 0.'

RESPONSE:
{
  "thought": "The user needs to dynamically pad a 2D array to the right and bottom sides using `np.pad`. The evidence indicates that the padding structure `((0, shape[0]-arr.shape[0]), (0, shape[1]-arr.shape[1]))` works correctly. I will write the final code block directly fitting inside function `f`.",
  "response": "    result = np.pad(arr, ((0, shape[0]-arr.shape[0]), (0, shape[1]-arr.shape[1])), 'constant')\n\n    return result"
}
"""
SOLVER_REWOO_FEW_SHOT_CONTRASTIVE_COT = """
=== FEW-SHOT EXAMPLE 1 ===
TASK:
Solve the following task or problem. To solve the problem, we have made step-by-step Plan and retrieved corresponding Evidence to each Plan.

	- Plan: 'Generate a Python function using pandas .iloc[] to reorder the DataFrame rows based on the provided list of positional indices.'
	- Evidence: 'Function logic using .iloc successfully generated and syntax verified.'
	- Plan: 'Verify the behavior of pandas .iloc[] with a list of integers in the documentation database to ensure no alignment side-effects occur.'
	- Evidence: 'DataFrame.iloc[list_of_ints] preserves absolute positional indexing correctly.'

Task: Problem:
I have the following DataFrame:
    Col1  Col2  Col3  Type
0      1     2     3     1
...
How can I achieve this?

[CORRECT TRAJECTORY]
{
  "thought": "The objective is to reorder rows of a pandas DataFrame using a sequence of position-based indices provided in a list. Based on the gathered evidence, the .iloc accessor is perfect for this as it handles pure integer position-based indexing. I will implement function `g(df, List)` that returns `df.iloc[List]` and then assign the final evaluation to `result`.",
  "response": "def g(df, List):\n    return df.iloc[List]\n\nresult = g(df.copy(), List)\n"
}

[WRONG TRAJECTORY]
{
  "thought": "I should use df.loc because it can search for elements, or maybe iterate through the rows with a for loop to reorder them manually.",
  "response": "result = pd.DataFrame()\nfor idx in List:\n    result = result.append(df.loc[idx])\n"
}

=== FEW-SHOT EXAMPLE 2 ===
TASK:
Solve the following task or problem. To solve the problem, we have made step-by-step Plan and retrieved corresponding Evidence to each Plan.

	- Plan: 'Ask the LLM to generate code using PyTorch broadcasting (unsqueeze and expand) and summation to calculate the batch dot product between `data` of shape (10, 6, 32) and `W` of shape (32,).'
	- Evidence: 'To align W (32,) with data (10, 6, 32), use W.unsqueeze(0).unsqueeze(0).expand(*data.size()) followed by torch.sum along dim 2.'
	- Plan: 'Search the code database to verify standard PyTorch implementations for batch dot products utilizing expand and sum to ensure optimization and correctness.'
	- Evidence: 'Confirmed: element-wise multiplication with an expanded tensor followed by sum over the feature dimension is a standard, efficient broadcasting pattern.'

Task: Problem:
I have batch data and want to dot() to the data. W is trainable parameters. How to dot between batch data and weights?
...
How to fix it?

[CORRECT TRAJECTORY]
{
  "thought": "The user encounters a shape mismatch using torch.bmm because W is 1D. Based on the evidence, the correct approach is to bring W to a 3D shape matching `data` by unsqueezing dimensions 0 and 1, expanding it to the size of `data`, performing element-wise multiplication, and then summing along the last dimension (axis 2). Finally, reshaping back to (10, 2, 3) as required.",
  "response": "W = W.unsqueeze(0).unsqueeze(0).expand(*data.size())\nresult = torch.sum(data * W, 2)\nresult = result.view(10, 2, 3)"
}

[WRONG TRAJECTORY]
{
  "thought": "I will just use torch.dot directly on the matrices or use torch.bmm by reshaping W to a matrix.",
  "response": "W = W.view(32, 1)\nresult = torch.bmm(data, W)"
}

=== FEW-SHOT EXAMPLE 3 ===
TASK:
Solve the following task or problem. To solve the problem, we have made step-by-step Plan and retrieved corresponding Evidence to each Plan.

	- Plan: 'Generate a Python function using `np.pad` to add zero padding dynamically to the bottom and right of a 2D array until it matches a target shape.'
	- Evidence: 'np.pad can perform asymmetric padding using the format ((top, bottom), (left, right)).'
	- Plan: 'Search the documentation database for zero padding examples using dynamic pad width calculations in NumPy to validate the padding schema.'
	- Evidence: 'The delta for bottom is target_shape[0] - arr.shape[0] and for right is target_shape[1] - arr.shape[1].'

Task: Problem:
I have a file with arrays or different shapes. I want to zeropad all the array to match the largest shape (93,13)...
Specifically, I want to pad to the right and bottom of original array in 2D.

[CORRECT TRAJECTORY]
{
  "thought": "The goal is to dynamically zero-pad a 2D array on the bottom and right edges to reach a target shape. Using np.pad with 'constant' mode is ideal. The padding structure must be ((0, shape[0] - arr.shape[0]), (0, shape[1] - arr.shape[1])). I will place this logic inside the requested function `f`.",
  "response": "    result = np.pad(arr, ((0, shape[0]-arr.shape[0]), (0, shape[1]-arr.shape[1])), 'constant')\n\n    return result\n"
}

[WRONG TRAJECTORY]
{
  "thought": "I can just create a new array of zeros with the target shape and copy the old array into the top-left corner using slicing, but np.pad with uniform width is easier.",
  "response": "    result = np.pad(arr, (93, 13), 'constant')\n    return result"
}
"""

FEW_SHOT_REGISTRY = {
    "zero_shot": "",
    "cot": REWOO_FEW_SHOT_COT_EXAMPLES,
    "contrastive_cot": REWOO_FEW_SHOT_CONTRASTIVE_COT,
    "auto_cot": REWOO_FEW_SHOT_AUTO_COT,
    "solver_cot": SOLVER_REWOO_FEW_SHOT_COT_EXAMPLES,
    "solver_contrastive_cot": SOLVER_REWOO_FEW_SHOT_CONTRASTIVE_COT,
    # "solver_auto_cot": SOLVER_REWOO_FEW_SHOT_AUTO_COT,
}

_LOG_SEPARATOR = f"\n{'_' * 20}\n"


def _create_step_model(tools: List[BaseTool]) -> Type[BaseModel]:
    """Create dynamic Step model with proper tool argument types."""
    tool_args_types = [tool.args for tool in tools if hasattr(tool, 'args') and tool.args is not None]
    
    if tool_args_types:
        ToolArgs = tool_args_types[0] if len(tool_args_types) == 1 else Union[tuple(tool_args_types)]
    else:
        ToolArgs = Dict[str, Any]
    
    return create_model(
        'Step',
        step_id=(int, ...),
        plan=(str, ...),
        tool=(str, ...),
        args=(ToolArgs, ...),
        evidence_tag=(str, ...),
        depends_on=(List[str], ...),
        __base__=BaseModel
    )


def _create_plan_model(tools: List[BaseTool]) -> Type[BaseModel]:
    """Create Plan model that uses dynamic Step models."""
    StepModel = _create_step_model(tools)
    return create_model('Plan', steps=(List[StepModel], ...), __base__=BaseModel)


class PlannerREWOOSGR(Agent):

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        temperature: float = 0.0,
        name: str = "rewoo_planner_agent",
        maximum_steps: int = 5,
        tools: List[BaseTool] = None,
        few_shot_type: str = "zero_shot"
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        self.tools = tools

        if not self.tools:
            self.tools = [LLMTool(url=url, model_name=model_name)]

        self.tool_names = ", ".join(tool.name for tool in self.tools)
        self.tools_dict = {t.name: t for t in self.tools}
        self.tools_prompt = "\n\n".join(
            [t.get_prompt_description() for t in self.tools]
        )

        self.prompt = PLANNER_PROMPT
        self.maximum_steps = maximum_steps

        if few_shot_type not in FEW_SHOT_REGISTRY:
            raise ValueError(f"Unknown few_shot_type: {few_shot_type}. Available: {list(FEW_SHOT_REGISTRY.keys())}")
            
        self.few_shot_examples = FEW_SHOT_REGISTRY[few_shot_type]

    def llm(self, prompt: str) -> BaseModel:
        """Call LLM to get a structured response."""
        Plan = _create_plan_model(self.tools)
        
        response = self.client.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format=Plan
        )

        return response.choices[0].message.parsed

    def run(self, task: Text) -> Dict[str, Any]:
        task_prompt = self.prompt.format(
            task=task,
            tool_names=self.tool_names,
            tools_formatted=self.tools_prompt
        )

        # parsed_response = self.llm(REWOO_FEW_SHOT_COT_EXAMPLES + task_prompt)
        parsed_response = self.llm(self.few_shot_examples + task_prompt)
        # parsed_response = self.llm(task_prompt)

        logging.info(f"TASK:\n\n{task}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"PLANNER TASK PROMPT:\n\n{task_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"PLAN PARSED RESPONSE:\n\n{parsed_response}")
        logging.info(_LOG_SEPARATOR)
        
        return parsed_response


class WorkerREWOOSGR(Agent):

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        name: str = "rewoo_worker_agent",
        tools: List[BaseTool] = None,
    ):
        super().__init__(name)
        self.tools = tools or [LLMTool(url=url, model_name=model_name)]
        self.tools_dict = {t.name: t for t in self.tools}
        self.worker_evidences = {}

    def _get_expected_args(self, tool: BaseTool) -> List[str]:
        """Get expected argument names for a tool."""
        if hasattr(tool, 'args') and tool.args is not None and hasattr(tool.args, '__annotations__'):
            return list(tool.args.__annotations__.keys())
        return []

    def execute_tool(self, tool_name: str, arguments:  Dict[str, Any]) -> str:
        """Execute a tool with given arguments."""
        if tool_name not in self.tools_dict:
            return f"unknown tool: {tool_name}"

        tool = self.tools_dict[tool_name]
        try:
            result = tool(**arguments)
            return str(result) if result is not None else ""
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"
    
    def _substitute_evidence_variables(self, tool_input: Union[Dict[str, Any], str], depends_on: List[str] = None) -> Union[Dict[str, Any], str]:
        """Replace evidence variables (#E1, #E2, etc.) with actual tool_result values.
        
        If depends_on is provided, only substitute variables that are in the depends_on list.
        This ensures that steps only access evidence they explicitly depend on.
        """
        if isinstance(tool_input, str):
            for var in re.findall(r"#E\d+", tool_input):
                if depends_on is None or var in depends_on:
                    if var in self.worker_evidences:
                        tool_input = tool_input.replace(
                            var, f"[{self.worker_evidences[var]['tool_result']}]"
                        )
            return tool_input
        elif isinstance(tool_input, dict):
            return {
                key: self._substitute_evidence_variables(value, depends_on) 
                for key, value in tool_input.items()
            }
        return tool_input

    def run(self, plan) -> Dict[str, Any]:
        """Execute the plan steps and collect evidence directly from Plan model."""
        for step in plan.steps:
            tool_name = step.tool
            tool_input = step.args.model_dump() if hasattr(step.args, "model_dump") else step.args
            
            logging.info(f"STEP ARGS:\n\n{step.args}")
            logging.info(_LOG_SEPARATOR)

            evidence_tag = step.evidence_tag

            tool_input = self._substitute_evidence_variables(tool_input, step.depends_on)
            
            logging.info(f"TOOL INPUT AFTER CALL:\n\n{tool_input}")
            logging.info(_LOG_SEPARATOR)

            tool_result = self.execute_tool(tool_name, tool_input)

            self.worker_evidences[evidence_tag] = {
                "tool": tool_name,
                "tool_input": tool_input,
                "tool_result": tool_result,
            }

        logging.info(f"WORKER EVIDENCES:\n\n{json.dumps(self.worker_evidences, indent=4)}")
        logging.info(_LOG_SEPARATOR)
        return self.worker_evidences

class SolverResponse(BaseModel):
    thought: str = Field(description="Internal chain of thought and logic analysis before writing the final solution.")
    response: str = Field(description="The final direct answer, code snippet, or solution with no extra conversational words.")

class SolverREWOOSGR(Agent):

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        name: str = "rewoo_solver_agent",
        temperature: float = 0.0,
        few_shot_type: str = "solver_cot"
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        self.prompt = SOLVER_PROMPT

        if few_shot_type not in FEW_SHOT_REGISTRY:
            raise ValueError(f"Unknown few_shot_type: {few_shot_type}. Available: {list(FEW_SHOT_REGISTRY.keys())}")
            
        self.few_shot_examples = FEW_SHOT_REGISTRY[few_shot_type]

    # def llm(self, prompt: str) -> str:
    #     """Сalling the llm to get a response."""
    #     response = self.client.chat.completions.create(
    #         model=self.model_name,
    #         messages=[{"role": "user", "content": prompt}],
    #         temperature=self.temperature,
    #     )
    #     return response.choices[0].message.content

    def llm(self, prompt: str) -> SolverResponse:
        """LLM calling"""
        response = self.client.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format=SolverResponse
        )
        return response.choices[0].message.parsed

    def _build_completed_plan_str(self, plan, evidencies) -> str:
        completed_plan = []
        for step in plan.steps:
            step_descr = step.plan
            evidence_tag = step.evidence_tag
            evidence_result = evidencies.get(evidence_tag, {}).get("tool_result", "No evidence available")
            completed_plan.append(f"\t- Plan: '{step_descr}'\n\t- Evidence: '{evidence_result}'")
        return "\n".join(completed_plan)
      
    # def run(self, task, plan, evidencies):
    #     # plan format: {"steps": matches, "plan_string": response}
    #     # evidence format: {"tool": tool, "tool_input": tool_input,"tool_result": tool_result}
    #     completed_plan_str = self._build_completed_plan_str(plan, evidencies)
    #     solve_prompt = self.prompt.format(plan=completed_plan_str, task=task)
        
    #     final_answer = self.llm(solve_prompt).strip()
        
    #     logging.info(f"SOLVE PROMPT:\n\n{solve_prompt}")
    #     logging.info(_LOG_SEPARATOR)
    #     logging.info(f"FINAL ANSWER:\n\n{final_answer}")
    #     logging.info(_LOG_SEPARATOR)
    #     return final_answer
      
    def run(self, task, plan, evidencies):
        completed_plan_str = self._build_completed_plan_str(plan, evidencies)
        solve_prompt = self.prompt.format(plan=completed_plan_str, task=task)
        
        full_prompt = self.few_shot_examples + "\n\n" + solve_prompt
        
        parsed_result = self.llm(full_prompt)
        
        logging.info(f"SOLVE PROMPT:\n\n{full_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"SOLVER INTERNAL THOUGHT:\n\n{parsed_result.thought}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"FINAL ANSWER (RESPONSE ONLY):\n\n{parsed_result.response}")
        logging.info(_LOG_SEPARATOR)
        
        return parsed_result.response

    def run_after_critic(self, task, plan, evidencies, critic):
        # plan format: {"steps": matches, "plan_string": response}
        # evidence format: {"tool": tool, "tool_input": tool_input,"tool_result": tool_result}
        completed_plan = []
        for step in plan.steps:
            step_descr = step.plan
            evidence_tag = step.evidence_tag
            evidence_result = evidencies.get(evidence_tag, {}).get("tool_result", "No evidence available")
            completed_plan.append(f"\t- Plan: '{step_descr}'\n\t- Evidence: '{evidence_result}'")
        
        completed_plan_str = "\n".join(completed_plan) + "\nCritic: " + critic
        solve_prompt = self.prompt.format(plan=completed_plan_str, task=task)
        
        final_answer = self.llm(solve_prompt).strip()
        
        logging.info(f"SOLVE PROMPT:\n\n{solve_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"FINAL ANSWER AFTER CRITIC:\n\n{final_answer}")
        logging.info(_LOG_SEPARATOR)
        return final_answer
