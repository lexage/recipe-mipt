"""Self-Refine prompts — CodeMMLU (multiple-choice) dataset.

CodeMMLU is multiple-choice: a task gives a Python code problem and four candidate options
(A, B, C, D); exactly one is reference-correct. The agent's answer is therefore a LETTER
(plus its reasoning), NOT a runnable implementation — so the FEEDBACK step critiques whether
the chosen option is reference-correct, and the (currently unused) REFINE step would output a
corrected letter; in the examples ``answer`` is the chosen letter + reasoning and ``refined``
is the corrected letter.

Mirrors the DS-1000 module and exposes the SAME public names (``FEEDBACK_INSTRUCTION``,
``REFINE_INSTRUCTION``, ``FEWSHOT_EXAMPLES``), so the agent switches datasets by import path
only. Aligned with the ReAct SOLVER gating: only concrete, verifiable reasons (behaviour
difference, failed example, wrong boundary, canonical form) should decide the answer —
style / naming / theoretical edge cases are ignored.

DATA-LEAKAGE NOTE: the few-shot examples below are ORIGINAL / SYNTHETIC. They are NOT drawn
from the CodeMMLU dataset, so no CodeMMLU item ids need to be excluded from metric counting.
"""

from .examples import Example

FEEDBACK_INSTRUCTION = """You are deciding a MULTIPLE-CHOICE question about Python code.
        You will be given a problem (with four options A, B, C, D) and a chosen answer
        (a letter and possibly its reasoning).
        Your goal is to write feedback that decides which option is reference-correct, for:
        1. Behaviour difference between an option's output and the expected behaviour
        2. A failed docstring / example, a wrong boundary, an off-by-one, or a raised error
        3. Logical errors in the chosen option's reasoning
        4. Conceptual misunderstandings of the problem
        5. When options are functionally equivalent, the canonical / minimal reference form
        Use ONLY concrete, verifiable reasons; do NOT raise issues of style, naming, or
        theoretical edge cases that do not change which option is reference-correct.
        Only provide a few sentence description, not an implementation, and finish with a
        single line:
        VERDICT: <letter> — <concrete reason>
        (confirm the chosen letter, or name the letter that is actually reference-correct)."""

REFINE_INSTRUCTION = """Now decide the reference-correct option using the feedback.
                                    Output EXACTLY ONE character — A, B, C, or D."""


# ---------------------------------------------------------------------------
# Few-shot examples — ORIGINAL / SYNTHETIC, NOT from CodeMMLU (no ids to exclude).
# `answer` = chosen letter + reasoning; `refined` = corrected letter.
# ---------------------------------------------------------------------------

example_1 = Example(
    question="""Choose the option whose function `is_prime(n)` correctly returns True iff n (n >= 2) is prime.
    A. for i in range(2, int(n ** 0.5)): if n % i == 0: return False ... return True
    B. for i in range(2, int(n ** 0.5) + 1): if n % i == 0: return False ... return True
    C. for i in range(2, n // 2): if n % i == 0: return False ... return True
    D. for i in range(2, n): if n % i == 0: return False ... return True""",
    answer="C — it stops the loop early at n // 2, so it looks efficient and correct.",
    feedback="""Option C is functionally correct but is NOT the canonical / minimal form, and
    option D (`range(2, n)`) is merely slower, not wrong. The decisive issue is the boundary in
    option A: it loops over `range(2, int(n ** 0.5))`, dropping the `+ 1`, so for n = 9 it never
    tests i = 3 (int(9 ** 0.5) == 3 and range(2, 3) yields only 2) and wrongly reports 9 as
    prime. Only option B uses the canonical `range(2, int(n ** 0.5) + 1)` bound with an early
    return, so the chosen answer C is not reference-correct.
    VERDICT: B — only B uses the canonical `int(n ** 0.5) + 1` bound and returns early.""",
    refined="B",
)
example_2 = Example(
    question="""Choose the option for `average(xs)` that returns the mean of xs and returns 0.0 for an empty
    list (docstring example: average([]) == 0.0).
    A. return sum(xs) / len(xs)
    B. if not xs: return 0; return sum(xs) / len(xs)
    C. if not xs: return 0.0; return sum(xs) / len(xs)
    D. return sum(xs) / len(xs) if xs[0] is not None else 0.0""",
    answer="A — sum divided by length is the definition of the mean.",
    feedback="""The function must return 0.0 on an empty list BEFORE dividing. The chosen option
    A computes `sum(xs) / len(xs)` with no guard, so for xs = [] it raises ZeroDivisionError
    instead of returning 0.0 — it fails the empty-input boundary in the docstring. Option B
    returns 0 (int, not the required 0.0) and option D indexes xs[0] before the guard, raising
    IndexError on []. Only option C guards with `if not xs: return 0.0` and then divides,
    matching every example including the empty case.
    VERDICT: C — C returns 0.0 on empty input before dividing; A raises ZeroDivisionError.""",
    refined="C",
)

FEWSHOT_EXAMPLES = [example_1, example_2]
