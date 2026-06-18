"""DeCRIM critique prompts — DS-1000 (code-generation) dataset.

Content extracted from the original inline prompts in ``decrim/main.py``. Exposes the
public names DECOMPOSE_PROMPT, CRITIQUE_PROMPT, CONSTRAINTS_HEADER so the agent can switch
datasets by changing only the import path (selected via the ``dataset`` param).

Templates go through ``str.format()``; any literal brace must be doubled ({{ }}).
"""

CONSTRAINTS_HEADER = "Constraints:"

DECOMPOSE_PROMPT = """You are an assistant whose job is to help me perform tasks.
I will give you a coding instruction that implicitly contains constraints to be followed.
Your task is to extract and list the constraints in a STRICT enumerated format.

FORMAT REQUIREMENTS:
- Use ONLY a numbered list where each line starts with a digit followed immediately by a dot and a space (e.g., "1. ", "2. ", "3. ").
- Each constraint must be on a separate line.
- Do NOT use markdown formatting, parentheses, dashes, Roman numerals, or any other numbering styles.
- Output ONLY the list itself, without introductory text, explanations, or concluding remarks.

Original Instruction: {question}

Provided Constraints:
"""

CRITIQUE_PROMPT = """You are an assistant whose job is to help me perform tasks.
I will give you a coding instruction and an AI assistant response.
The response should be a valid and correct piece of code which follows the syntax of the programming language.
The instruction includes some constraints to be followed by AI assistant while generating response.
Your task is to check and let me know which of the constraints are satisfied by the AI assistant response.
Please state short reasons on whether constraint is satisfied in the response or not.
Also include final answer as "Constraint followed" or "Constraint not followed" for each constraint.

Instruction: {question}

{constraints_text}

Assistant Response: {answer}

Please analyze each constraint one by one and provide your critique:
"""
