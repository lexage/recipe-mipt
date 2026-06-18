"""CRITIC prompts — CodeMMLU (multiple-choice) dataset.

CodeMMLU is multiple-choice: a task gives a Python code problem and four candidate options
(A, B, C, D); exactly one is reference-correct. The agent's answer is a LETTER (plus its
reasoning), NOT a runnable implementation.

The CRITIC idea is reinterpreted for MCQ (per the agent's ``_gather_evidence_mc``):
1. extract the code of EACH option (``GET_OPTIONS_PROMPT``);
2. execute each option to gather behavioural evidence (python_repl tool);
3. compare the options' outputs against each other and the problem's expected behaviour
   (``MC_COMPARE_PROMPT``);
4. critique: explain why three options are wrong and which one is reference-correct, ending
   with a single VERDICT letter (``CRITIQUE_SYSTEM_PROMPT`` / ``CRITIQUE_HUMAN_PROMPT``).

Mirrors the DS-1000 module and exposes the SAME public names, plus the MC-only extras
``GET_OPTIONS_PROMPT`` and ``MC_COMPARE_PROMPT``. Aligned with the ReAct SOLVER gating:
only concrete, verifiable reasons (behaviour difference, failed example, wrong boundary,
canonical form) should decide the answer — style/naming/theoretical edge cases are ignored.

Many CodeMMLU items are conceptual (not runnable); the orchestration degrades gracefully and
the prompts must still produce a verdict from reasoning alone when execution yields no signal.

Templates filled via ``str.format()`` (GET_OPTIONS_PROMPT, MC_COMPARE_PROMPT, COMPARE_PROMPT)
or fed to a langchain ChatPromptTemplate (CRITIQUE_* — keep only the documented braces).
Any other literal brace must be doubled ({{ }}).
"""

# Extract the runnable code of EACH option. Output strictly delimited so it can be parsed
# back into a {letter: code} map (see Critic._parse_options_response). The [[X]] markers
# precede each option's code; emit an option with empty body if it carries no runnable code.
GET_OPTIONS_PROMPT = (
    "Below is a MULTIPLE-CHOICE Python question with four options labelled A, B, C, D.\n"
    "Extract the code of each option so it can be executed independently.\n\n"
    "OUTPUT FORMAT (strict):\n"
    "- For each option output a line containing only its marker [[A]], [[B]], [[C]] or [[D]],\n"
    "  immediately followed by the option's runnable Python code on the next lines.\n"
    "- Include the minimal surrounding code (function definition plus a call that prints the\n"
    "  result) needed to actually run the option, when the problem implies one.\n"
    "- If an option has no runnable code (purely conceptual statement), leave its body empty.\n"
    "- Do NOT add explanations, markdown fences, or any text outside the markers and code.\n\n"
    "Problem (with options A/B/C/D): {problem}\n"
)

# Compare the per-option execution results against each other and the expected behaviour.
MC_COMPARE_PROMPT = (
    "You are deciding a MULTIPLE-CHOICE Python question. Below are the execution results of "
    "each option (A/B/C/D) and the original problem describing the expected behaviour. "
    "Compare the options' outputs against each other and against the expected behaviour from "
    "the problem. State, per option, whether its observed behaviour matches what the problem "
    "requires, citing the concrete output or error. Return only the comparison.\n\n"
    "Problem (with options A/B/C/D): {problem}\n\n"
    "Per-option execution results:\n{options_execution}\n"
)

# Same signature as DS-1000 so the langchain compare_tool stays uniform across datasets;
# here "implementation" is the chosen option (letter / its code) and the comparison is to
# the expected behaviour stated in the problem.
COMPARE_PROMPT = (
    "You will get a multiple-choice problem, a chosen option, and the result of compiling that "
    "option. Compare the expected behaviour described in the problem with the compilation result. "
    "Return only result of comparison. "
    "Problem: {problem} \n Implementation: {implementation} \n compilation_result: {compilation_result}  "
)

# System instruction for the critique step (no template variables).
CRITIQUE_SYSTEM_PROMPT = """You are deciding a MULTIPLE-CHOICE question about Python code.
You will be given the problem (with four options A/B/C/D), an INITIAL ANSWER from another agent
(a chosen letter, possibly with its reasoning), and behavioural evidence: the execution result
of each option and a comparison against the problem's expected behaviour.

Generate criticism that decides which option is reference-correct, using only concrete,
verifiable reasons grounded in the evidence:
1. Behaviour difference between an option's output and the expected behaviour
2. A failed docstring / example, a wrong boundary, an off-by-one, or a raised error
3. Logical errors in the option's reasoning
4. Conceptual misunderstandings of the problem
5. When options are functionally equivalent, the canonical / minimal reference form
Do NOT raise issues of style, naming, or theoretical edge cases that do not change which
option is reference-correct. If execution produced no usable signal (conceptual question),
decide from the problem and the options' code by reasoning alone.

Explain briefly why each WRONG option is wrong, then finish with a single line:
VERDICT: <letter> — <concrete reason>
(confirm the initial letter, or name the letter that is actually reference-correct).
Return only the criticism and the verdict, not an implementation."""

# Human turn for the critique step (langchain template variables). Here `implementation` is
# the agent's initial answer (chosen letter + reasoning) and `tools_result` is the per-option
# execution evidence and comparison.
CRITIQUE_HUMAN_PROMPT = (
    "Problem (with options A/B/C/D): {problem} \n Initial Answer: {implementation} \n "
    "Evidence: {tools_result}  . "
)
