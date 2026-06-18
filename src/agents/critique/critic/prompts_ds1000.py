"""CRITIC prompts — DS-1000 (code-generation) dataset.

Content extracted verbatim from the original inline prompts in ``critic/main.py``.
Exposes the public names the agent reads via ``self.prompts.*`` so switching benchmarks
changes only the imported module (selected by the ``dataset`` param).

Code-generation flow: the agent answer IS a Python implementation — it is extracted
(``GET_CODE_PROMPT``), compiled, its output compared to the expected behaviour
(``COMPARE_PROMPT``), and finally critiqued (``CRITIQUE_SYSTEM_PROMPT`` / ``CRITIQUE_HUMAN_PROMPT``).

Templates filled via ``str.format()`` (GET_CODE_PROMPT, COMPARE_PROMPT) or fed to a
langchain ChatPromptTemplate (CRITIQUE_* — keep ``{problem}``/``{implementation}``/``{tools_result}``
as the only braces). Any other literal brace must be doubled ({{ }}).
"""

# Extract the runnable Python implementation from the agent's answer.
GET_CODE_PROMPT = (
    "You only need to extract the Python code from the following text. "
    "I want to compile this code. And You can't fix it. "
    "If you can't find python code, return nothing. Text: {answer} "
)

# Compare the expected result of the problem against the actual compilation result.
COMPARE_PROMPT = (
    "You will get a problem, implementation and compilation result of this implementation. "
    "You should compare the expected result of problem and compilation's result. "
    "Return only result of comparison. "
    "Problem: {problem} \n Implementation: {implementation} \n compilation_result: {compilation_result}  "
)

# System instruction for the critique step (no template variables).
CRITIQUE_SYSTEM_PROMPT = """You will be given a problem and Implementation in Python and information about Implementation.
You should generate criticism of this Implementation using the information:
It is important for you to use following criteria:
1. Evaluation of implementation
2. Logical errors in reasoning
3. Syntax and semantic correctness
4. Conceptual misunderstandings
5. Potential bugs or edge cases
6. Alignment with problem requirements You will need this as a hint when you
 Return only Criticism, not implementation"""

# Human turn for the critique step (langchain template variables).
CRITIQUE_HUMAN_PROMPT = (
    "Problem: {problem} \n Implementation: {implementation} \n Information: {tools_result}  . "
)
