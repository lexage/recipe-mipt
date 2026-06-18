"""Prompt templates and few-shot examples for the ReWOO pipeline (Planner/Worker/Solver).

Extracted from rewoo.py to keep the agent logic readable.
Imported back via `from src.agents.pipelines.prompts.ds1000.prompts_rewoo import ...`.
"""

PLANNER_PROMPT = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Available tool names only: {tool_names}

{tools_formatted}

For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked. #E2 = llm[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = calculator[(2 ∗ #E2 − 10) − 8]

Begin!
Describe your plans with rich details. Each Plan should be followed by only one #E.

Task: {task}"""


WORKER_PROMPT = """Respond in short directly with no extra words.\n\n{request}"""


SOLVER_PROMPT = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""
