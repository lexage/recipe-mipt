"""Reflexion prompts — CodeMMLU (multiple-choice) dataset.

CodeMMLU is multiple-choice: a task gives a Python code problem and four candidate options
(A, B, C, D); exactly one is reference-correct. The agent's answer is therefore a LETTER
(plus its reasoning), NOT a runnable implementation.

The Reflexion idea is reinterpreted for MCQ but the agent's two-step pipeline is unchanged
(``evaluate`` -> ``reflect``); only the prompts differ from the DS-1000 module:
1. evaluate — score 1..5 how strongly the CHOSEN option is the reference-correct one, judged
   on concrete behaviour (``PY_EVALUATE_INSTRUCTION`` / ``PY_EVALUATE_FEW_SHOT``);
2. reflect — verbal feedback / hint for the next attempt, grounded only in concrete,
   verifiable reasons (behaviour difference, failed example, wrong boundary, canonical form),
   ending with a single ``VERDICT: <letter> — <reason>`` line
   (``PY_SELF_REFLECTION_INSTRUCTION`` / ``PY_SELF_REFLECTION_FEW_SHOT``).

Mirrors the DS-1000 module and exposes the SAME public names, so the agent switches datasets
by import path only. Aligned with the ReAct SOLVER gating: only concrete, verifiable reasons
should decide the answer — style / naming / theoretical edge cases are ignored.

DATA-LEAKAGE NOTE: the few-shot examples below are ORIGINAL / SYNTHETIC. They are NOT drawn
from the CodeMMLU dataset, so no CodeMMLU item ids need to be excluded from metric counting.
"""

PY_SELF_REFLECTION_INSTRUCTION = """You are deciding a MULTIPLE-CHOICE question about Python code.
        You will be given a problem (with four options A, B, C, D), a chosen answer (a letter and
        possibly its reasoning), and a numeric evaluation of that choice.
        Your goal is to write a short feedback / hint for the next attempt, considering:
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

PY_EVALUATE_INSTRUCTION = """You are deciding a MULTIPLE-CHOICE question about Python code.
        You will be given a problem (with four options A, B, C, D) and a chosen answer
        (a letter, possibly with its reasoning).
        Your goal is to compute a reward score that reflects how strongly the chosen option is
        the reference-correct one, considering:
        1. Behaviour difference between an option's output and the expected behaviour
        2. A failed docstring / example, a wrong boundary, an off-by-one, or a raised error
        3. Logical errors in the chosen option's reasoning
        4. Conceptual misunderstandings of the problem
        5. When options are functionally equivalent, whether the chosen option is the canonical
        / minimal reference form
        A score of 5 means the chosen option clearly satisfies every example and boundary; a low
        score means another option is reference-correct.
        You only need to return a score as an integer from 1 to 5."""

PY_ACTOR_INSTRUCTION = (
    "You are deciding a multiple-choice question about Python code. "
    "You will be given a problem with four options A, B, C, D and a reflection on a previous "
    "choice. Your goal is to pick the reference-correct option using the reflection. "
    "Output EXACTLY ONE character — A, B, C, or D."
)


# ---------------------------------------------------------------------------
# Few-shot examples — ORIGINAL / SYNTHETIC, NOT from CodeMMLU (no ids to exclude).
# ---------------------------------------------------------------------------

SELF_REFLECTION_1 = """Option B uses `n % i == 0` over `range(2, int(n ** 0.5) + 1)` and returns
       False as soon as a divisor is found, which is the correct primality test. The chosen
       option C stops the loop at `range(2, n // 2)`, which is functionally correct for the
       result but is NOT the canonical / minimal form — and option D uses `range(2, n)` which is
       merely slower, not wrong. The decisive issue is the boundary in option A: it loops over
       `range(2, int(n ** 0.5))`, dropping the `+ 1`, so for n = 9 it never tests i = 3
       (int(9 ** 0.5) == 3, and range(2, 3) yields only 2) and wrongly reports 9 as prime.
       The chosen answer C is therefore not reference-correct; the canonical correct option is B.
       VERDICT: B — only B uses the canonical `int(n ** 0.5) + 1` bound and returns early."""

SELF_REFLECTION_2 = """The function must return the average, and on an empty list return 0.0
       BEFORE dividing. The chosen option A computes `sum(xs) / len(xs)` with no guard, so for
       xs = [] it raises ZeroDivisionError instead of returning 0.0 — it fails the empty-input
       boundary stated in the docstring. Option C guards with `if not xs: return 0.0` and then
       divides, matching every example including the empty case. Options B and D either return 0
       (int, not the required 0.0) or guard after an indexing access. The chosen answer A is
       wrong on the empty-list boundary.
       VERDICT: C — C returns 0.0 on empty input before dividing; A raises ZeroDivisionError."""

PY_SELF_REFLECTION_FEW_SHOT = f"""
    Example 1:
    Problem:
    Choose the option whose function `is_prime(n)` correctly returns True iff n (n >= 2) is prime.
    A. for i in range(2, int(n ** 0.5)): if n % i == 0: return False ... return True
    B. for i in range(2, int(n ** 0.5) + 1): if n % i == 0: return False ... return True
    C. for i in range(2, n // 2): if n % i == 0: return False ... return True
    D. for i in range(2, n): if n % i == 0: return False ... return True

    Implementation:
    C — it stops the loop early at n // 2, so it looks efficient and correct.

    Evaluation: 2

    Reflection:
    {SELF_REFLECTION_1}

    Example 2:
    Problem:
    Choose the option for `average(xs)` that returns the mean of xs and returns 0.0 for an empty
    list (docstring example: average([]) == 0.0).
    A. return sum(xs) / len(xs)
    B. if not xs: return 0; return sum(xs) / len(xs)
    C. if not xs: return 0.0; return sum(xs) / len(xs)
    D. return sum(xs) / len(xs) if xs[0] is not None else 0.0

    Implementation:
    A — sum divided by length is the definition of the mean.

    Evaluation: 2

    Reflection:
    {SELF_REFLECTION_2}
    END OF EXAMPLES
"""


PY_EVALUATE_FEW_SHOT = """
    Example 1:
    Problem:
    Choose the option whose function `is_prime(n)` correctly returns True iff n (n >= 2) is prime.
    A. for i in range(2, int(n ** 0.5)): if n % i == 0: return False ... return True
    B. for i in range(2, int(n ** 0.5) + 1): if n % i == 0: return False ... return True
    C. for i in range(2, n // 2): if n % i == 0: return False ... return True
    D. for i in range(2, n): if n % i == 0: return False ... return True

    Implementation:
    C — it stops the loop early at n // 2, so it looks efficient and correct.

    Evaluation:
    2


    Example 2:
    Problem:
    Choose the option for `average(xs)` that returns the mean of xs and returns 0.0 for an empty
    list (docstring example: average([]) == 0.0).
    A. return sum(xs) / len(xs)
    B. if not xs: return 0; return sum(xs) / len(xs)
    C. if not xs: return 0.0; return sum(xs) / len(xs)
    D. return sum(xs) / len(xs) if xs[0] is not None else 0.0

    Implementation:
    A — sum divided by length is the definition of the mean.

    Evaluation:
    2

    END OF EXAMPLES
"""
