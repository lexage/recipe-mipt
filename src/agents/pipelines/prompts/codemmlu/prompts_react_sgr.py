"""Prompt templates for ReActAgentSGR — CodeMMLU dataset.

CodeMMLU is a multiple-choice benchmark: each task gives a Python code problem and
four candidate options (A, B, C, D); exactly one is the reference-correct answer.
Two sub-types are covered:
  * code_completion   — a function (docstring + examples) and four completions;
                        failures are mostly REAL bugs -> reason on edge cases.
  * fill_in_the_middle — a fragment with a missing line and four options that are
                        often FUNCTIONALLY EQUIVALENT -> pick the canonical form.

Mirrors the DS-1000 file at `prompts.ds1000.prompts_react_sgr` and exposes the SAME
public names (REACT_SYSTEM_PROMPT, FINISH_PROMPT_TEMPLATE, SOLVER_PROMPT,
FEW_SHOT_REGISTRY), so the pipeline can switch datasets by changing only the import
path. Imported back via `from src.agents.pipelines.prompts.codemmlu.prompts_react_sgr import ...`.

Few-shot examples are intentionally not provided yet — FEW_SHOT_REGISTRY keeps the
same keys as the DS-1000 version (so `few_shot_type` validation still passes) with
empty values.

NOTE on braces: REACT_SYSTEM_PROMPT, FINISH_PROMPT_TEMPLATE and SOLVER_PROMPT are
passed through str.format(), so any literal brace must be doubled ({{ }}).
"""

REACT_SYSTEM_PROMPT = """You are an autonomous ReAct agent answering a MULTIPLE-CHOICE question about Python code.
You are given a code problem and four candidate options labelled A, B, C, D. EXACTLY ONE is the
reference-correct answer. Interact with tools to decide which letter it is. Minimize steps; finish
immediately when the answer is known.

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

HOW TO DECIDE THE LETTER
STEP 1 — Reject any option that changes BEHAVIOUR. Mentally run each option on the cases the
problem implies (the docstring examples, and the boundaries: empty input, a single element, 0, 1,
negatives, duplicates, first/last index). Reject a wrong comparison/operator/variable, an off-by-one
(e.g. a primality bound that must reach int(n**0.5)+1; '>' vs '>=', '<0' vs '<=0' where the result
changes), parentheses that change precedence (Python: `and` binds tighter than `or`, so
`a and b or c` != `a and (b or c)`), a dropped guard the spec needs (empty list -> return the
sentinel such as nan / [] / None BEFORE dividing or indexing), data that must keep duplicates being
deduplicated, a wrong formula/tie-break, or a different algorithm. Keep the option whose result
matches EVERY example exactly.
STEP 2 — If two or more surviving options are FUNCTIONALLY EQUIVALENT (identical result on every
input), pick the CANONICAL / minimal form — the one closest, character for character, to the
simplest reference form: midpoint `(l + r) // 2` (not `l+(r-l)//2`, `(l+r)>>1`); `math.inf` (not
`float('inf')`, `sys.maxsize`); `[x] * n` (not `[x for _ in range(n)]`, `n*[x]`); bare tuple
`return a, b` (not `return [a, b]`); `1 + max(...)` (not `max(...) + 1`); `3 * (row // 3)` (not
`(row // 3) * 3`); strict `>` to track a maximum, keeping the FIRST among equal values; no extra
guard/cast/paren the spec does not need; PEP 8 spacing `[1] * (n + 1)` (not `[1]*(n+1)`). Judge by
the FORM, not by the position — the canonical option is frequently A; never reject or avoid an
option just because of where it appears.

CONSTRAINTS
1. Action MUST be one of: {tool_names} or 'finish'.
   IMPORTANT: Few-shot examples may demonstrate different tools for format/structure reference only.
   You must strictly call ONLY the available tools listed in {tool_names}.
2. action_input MUST match the tool's argument schema exactly.
3. If a tool error occurs, correct the arguments and retry once.

FINALIZATION
When action='finish':
- Set is_final=true.
- Your reasoning must have settled on EXACTLY ONE letter (A, B, C, or D). State it in 'thought'.
"""


FINISH_PROMPT_TEMPLATE = """You are providing the FINAL ANSWER to a multiple-choice code question.

CONVERSATION HISTORY:
{history_text}

TASK:
{task}

INSTRUCTIONS:
- Decide which single option is reference-correct based on the reasoning above.
- Output EXACTLY ONE character — A, B, C, or D. No words, no code, no punctuation, no explanation.
"""


SOLVER_PROMPT = """You are a precise Solver Agent for a MULTIPLE-CHOICE code question. You produce the final
answer by reconciling an initial choice with explicit critic feedback.

INPUTS:
TASK (problem + options A/B/C/D): {task}
INITIAL_ANSWER: {answer}
CRITIC_FEEDBACK: {critic}

INSTRUCTIONS:
1. The answer is a SINGLE letter — A, B, C, or D. There is exactly one reference-correct option.
2. CRITIC GATING: Change the INITIAL_ANSWER only if the CRITIC_FEEDBACK gives a concrete, verifiable
   reason that another option is correct (a behaviour difference, a failed example, a wrong
   boundary). Ignore feedback about style, naming, or theoretical edge cases that do not change
   which option is reference-correct.
3. TIE-BREAK: If the surviving options are functionally equivalent, keep the canonical / minimal
   form (the simplest standard idiom), judging by FORM not position.
4. If the feedback is vague or merely confirms the initial choice, keep the INITIAL_ANSWER.

OUTPUT RULES:
- Output EXACTLY ONE character — A, B, C, or D.
- ZERO meta-commentary, no references to the critique, no reasoning steps, no punctuation.

FINAL ANSWER:
"""


# =====================================================================================
# FEW-SHOT EXAMPLES
# =====================================================================================
# Each TASK below is produced exactly the way `build_agent_task()` assembles a live task
# (see codemmlu_agent_pipelines.py): a "Problem: {input}" line followed by the four
# "Solution A/B/C/D" options. The only difference between the two sub-types is `input`:
#   * code_completion   -> input = question
#   * fill_in_the_middle -> input = question + " " + problem_description
# The TASKs are taken verbatim from examples.json (3 code_completion + 3 fill_in_the_middle).
#
# Trajectories use the same readable Thought/Action/Observation shape as the DS-1000
# sibling file. They may reference the `db_search`/`llm` tools for FORMAT reference only —
# at run time the agent must call ONLY the tools actually registered (see the system
# prompt CONSTRAINTS section).
# -------------------------------------------------------------------------------------


# ---- code_completion (distractors are usually REAL bugs -> reason on edge cases) ----
FEW_SHOT_COT_CODE = """### CODE_COMPLETION FEW-SHOT EXAMPLES ###

=== FEW-SHOT EXAMPLE 1 ===
TASK:
Problem: from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"

Solution A:   for i in range(len(numbers) - 1):
    for j in range(i + 1, len(numbers)):
      if abs(numbers[i] - numbers[j]) > threshold:
        return False
  return True
Solution B:   return any(abs(a - b) < threshold for a, b in zip(numbers, numbers[1:]))
Solution C:   for i in range(len(numbers)):  # Change range to len(numbers)
    for j in range(i + 1, len(numbers)):
      if abs(numbers[i] - numbers[j]) < threshold:
        return True
  return False
Solution D:     for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False

Thought 1:
The function must return True iff SOME pair is closer than `threshold`. I trace the docstring cases: [1.0,2.0,3.0],0.5 -> False; [1.0,2.8,3.0,...],0.3 -> True (2.8 vs 3.0 differ by 0.2 < 0.3). Now I reject options that change behaviour. A returns `False` on the first pair with distance `> threshold`, i.e. it fires on a FAR pair — inverted logic, so on [1.0,2.0,3.0] it returns False on (1.0,2.0) although no pair is close: WRONG. B only compares `zip(numbers, numbers[1:])`, i.e. ADJACENT pairs, so it misses close pairs that are not neighbours: WRONG. C and D both scan pairs and return True on `< threshold`, else False — both are correct and functionally EQUIVALENT. This is a genuine tie, so I keep the canonical reference form: the full pairwise double loop over enumerate (D). Let me confirm the canonical shape.

Action 1: db_search
Action Input 1: {'query': 'return true if any two elements are closer than threshold pairwise nested loop enumerate'}
Is_final 1: False

Observation 1: query: return true if any two elements are closer than threshold pairwise nested loop enumerate
retrieved context: [CHUNK 2101 | doc=0071]
def has_close_elements(numbers, threshold):
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2 and abs(elem - elem2) < threshold:
                return True
    return False

Thought 2:
The retrieved reference is the full pairwise enumerate scan with the `idx != idx2` guard — identical to option D. C is equivalent but D is the canonical reference form. The answer is D.

Action 2: finish
Action Input 2: {}
Is_final 2: True

=== FEW-SHOT EXAMPLE 2 ===
TASK:
Problem:
def generate_integers(a, b):
    \"\"\"
    Given two positive integers a and b, return the even digits between a
    and b, in ascending order.

    For example:
    generate_integers(2, 8) => [2, 4, 6, 8]
    generate_integers(8, 2) => [2, 4, 6, 8]
    generate_integers(10, 14) => []
    \"\"\"

Solution A:     lower = max(2, min(a, b))
    upper = min(8, max(a, b))

    return [i for i in range(lower, upper+1) if i % 2 != 0]
Solution B:     lower = max(2, min(a, b))
    upper = min(8, max(a, b))

    return [i for i in range(lower, upper+1) if i % 2 == 0]
Solution C:     lower = max(2, min(a, b))
    upper = min(8, max(a, b))

    return [i for i in range(lower, upper) if i % 2 == 0]
Solution D:     lower = min(2, min(a, b))
    upper = max(8, max(a, b))

    return [i for i in range(lower, upper+1) if i % 2 == 0]

Thought 1:
The spec asks for the EVEN one-digit values in [2, 8] that lie between a and b, ascending. Trace: generate_integers(2, 8) => [2, 4, 6, 8]; generate_integers(10, 14) => [] (no even digit in 2..8 lies in 10..14). I reject behaviour changes. A filters `i % 2 != 0`, returning ODD values -> [3, 5, 7] for (2, 8): WRONG. C iterates `range(lower, upper)`, excluding `upper`, so it drops 8 for (2, 8) -> [2, 4, 6]: off-by-one, WRONG. D clamps with `min(2, ...)` / `max(8, ...)`, which does NOT bound the interval to [2, 8]; for (10, 14) it yields even numbers up to 14: WRONG. B clamps to [2, 8] with `max(2, ...)` / `min(8, ...)`, iterates inclusively, keeps `i % 2 == 0`. Let me confirm the inclusive-range boundary.

Action 1: db_search
Action Input 1: {'query': 'inclusive range clamp lower upper keep even numbers list comprehension'}
Is_final 1: False

Observation 1: query: inclusive range clamp lower upper keep even numbers list comprehension
retrieved context: [CHUNK 3390 | doc=0142]
lo, hi = max(2, min(a, b)), min(8, max(a, b))
[i for i in range(lo, hi + 1) if i % 2 == 0]   # inclusive of hi; keeps evens

Thought 2:
The reference clamps to [2, 8], uses an inclusive `range(lo, hi + 1)`, and keeps even values — exactly option B. The answer is B.

Action 2: finish
Action Input 2: {}
Is_final 2: True

=== FEW-SHOT EXAMPLE 3 ===
TASK:
Problem:
def prime_length(string):
    \"\"\"Write a function that takes a string and returns True if the string
    length is a prime number or False otherwise
    Examples
    prime_length('Hello') == True
    prime_length('abcdcba') == True
    prime_length('kittens') == True
    prime_length('orange') == False
    \"\"\"

Solution A:     l = len(string)
    if l == 0 or l == 1:
        return False
    return l % 2 != 0
Solution B:     l = len(string)
    if l == 0 or l == 1:
        return False
    for i in range(2, l):
        if l % i == 0:
            return False
    return True
Solution C:     l = len(string)
    if l == 0 or l == 1:
        return False
    return l % 3 != 0
Solution D:     l = len(string)
    if l == 0 or l == 1:
        return False
    return l % 2 == 1

Thought 1:
The function must return True iff `len(string)` is prime. A and D are the SAME test: `l % 2 != 0` and `l % 2 == 1` both just check ODDness, so length 9 (odd, not prime) returns True — WRONG for both. C uses `l % 3 != 0`, which is unrelated to primality (length 25 -> True though 25 is not prime): WRONG. B is a real primality test by trial division over `range(2, l)`, returning False on any divisor. Let me confirm the trial-division form.

Action 1: db_search
Action Input 1: {'query': 'primality test trial division for i in range 2 to n return false if divisible'}
Is_final 1: False

Observation 1: query: primality test trial division for i in range 2 to n return false if divisible
retrieved context: [CHUNK 4517 | doc=0203]
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

Thought 2:
Option B is the canonical trial-division primality test; the odd/mod-3 shortcuts (A, C, D) are not real primality checks. The answer is B.

Action 2: finish
Action Input 2: {}
Is_final 2: True
"""


# ---- fill_in_the_middle (options are usually EQUIVALENT -> pick the canonical form) ----
FEW_SHOT_COT_MIDDLE = """### FILL_IN_THE_MIDDLE FEW-SHOT EXAMPLES ###

=== FEW-SHOT EXAMPLE 1 ===
TASK:
Problem: class Solution:
  def dominantIndex(self, nums: List[int]) -> int:
    max = 0
    secondMax = 0

    for i, num in enumerate(nums):
      if num > max:
        secondMax = max
        max = num
        ans = i
      elif num > secondMax:
        secondMax = num

    _______________
    return ans if max >= 2 * secondMax else -1 You are given an integer array nums where the largest integer is unique. Determine whether the largest element in the array is at least twice as much as every other number in the array. If it is, return the index of the largest element, or return -1 otherwise.

Solution A:     return ans if max > 2 * secondMax else -1
Solution B:     return ans if max >= 2 * secondMax else -1
Solution C:     return ans if max == 2 * secondMax else -1
Solution D:     return ans if max // 2 >= secondMax else -1

Thought 1:
The spec says "at least twice as much", i.e. `max >= 2 * secondMax`; the reference line already shown after the blank is exactly that. So the fill line must reproduce the canonical `>=` comparison. I reject the behaviour-changers. A uses `>`, which drops the equality boundary and returns -1 when the max is EXACTLY twice the runner-up — WRONG ("at least" includes equality). C uses `==`, which is only true when max is exactly twice — WRONG. D rewrites it as `max // 2 >= secondMax`; integer division discards the remainder and is a non-canonical reformulation that can disagree on odd values — reject in favour of the direct form. B is the canonical `>=` expression.

Action 1: db_search
Action Input 1: {'query': 'at least twice as large comparison canonical greater than or equal 2 times'}
Is_final 1: False

Observation 1: query: at least twice as large comparison canonical greater than or equal 2 times
retrieved context: [CHUNK 5120 | doc=0311]
# "x is at least twice y"  ->  x >= 2 * y      (canonical; keeps the equality case)

Thought 2:
"At least twice" maps to the canonical `max >= 2 * secondMax`, which is option B. The answer is B.

Action 2: finish
Action Input 2: {}
Is_final 2: True

=== FEW-SHOT EXAMPLE 2 ===
TASK:
Problem: class Solution:
  def makeLargestSpecial(self, s: str) -> str:
    specials = []
    count = 0

    i = 0
    for j, c in enumerate(s):
      count += 1 if c == '1' else -1
      if count == 0:
        specials.append(
            _______________
            '1' + self.makeLargestSpecial(s[i + 1:j]) + '0')
        i = j + 1

    return ''.join(sorted(specials)[::-1]) Special binary strings are binary strings with the following two properties: You are given a special binary string s. A move consists of choosing two consecutive, non-empty, special substrings of s, and swapping them. Two strings are consecutive if the last character of the first string is exactly one index before the first character of the second string. Return the lexicographically largest resulting string possible after applying the mentioned operations on the string.

Solution A:             self.makeLargestSpecial(s[i + 1:j]) + '10'
Solution B:             '10' + self.makeLargestSpecial(s[i + 1:j]) + '0'
Solution C:             '1' + self.makeLargestSpecial(s[i + 1:j]) + '01'
Solution D:             '1' + self.makeLargestSpecial(s[i + 1:j]) + '0')

Thought 1:
Each balanced block is rebuilt by wrapping the recursively-normalised inner substring as `'1' + makeLargestSpecial(inner) + '0'`. The line already shown after the blank is `'1' + self.makeLargestSpecial(s[i + 1:j]) + '0')`, so the fill must reproduce that exact wrapping and close the `append(` call. A appends `... + '10'` (drops the leading `'1'`, mangles the wrap): WRONG. B prepends `'10'` and appends `'0'`: WRONG. C appends `... + '01'` instead of `'0'`: WRONG. D is `'1' + self.makeLargestSpecial(s[i + 1:j]) + '0')` — the canonical `'1'...'0'` wrap that also closes the parenthesis. The answer is D.

Action 1: finish
Action Input 1: {}
Is_final 1: True

=== FEW-SHOT EXAMPLE 3 ===
TASK:
Problem: class Solution:
  def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
    while sx < tx and sy < ty:
      tx, ty = tx % ty, ty % tx

    _______________
    return sx == tx and sy <= ty and (ty - sy) % tx == 0 or \\
        sy == ty and sx <= tx and (tx - sx) % ty == 0 Given four integers sx, sy, tx, and ty, return true if it is possible to convert the point (sx, sy) to the point (tx, ty) through some operations, or false otherwise. The allowed operation on some point (x, y) is to convert it to either (x, x + y) or (x + y, y).

Solution A:     return sx == tx and sy < ty and (ty + sy) % tx == 0 or \\
Solution B:     return sx == tx and sy <= ty and (sx - sy) % tx == 0 or \\
Solution C:     return sx == tx and sy <= ty and (ty - sy) % tx == 0 or \\
Solution D:     return sx == tx and sy < ty and (ty - sy) % tx == 0 or \\

Thought 1:
The blank is the FIRST physical line of the final two-line return; the second line `sy == ty and sx <= tx and (tx - sx) % ty == 0` is already shown, so the fill must be its symmetric counterpart: `sx == tx and sy <= ty and (ty - sy) % tx == 0`. Reject the deviations. A uses `sy < ty` (drops the equality boundary) and `(ty + sy)` (wrong operator): WRONG. B computes `(sx - sy) % tx`, the wrong operands: WRONG. D uses `sy < ty` instead of `sy <= ty`, dropping the boundary where sy equals ty: WRONG. C matches the canonical symmetric line exactly. The answer is C.

Action 1: finish
Action Input 1: {}
Is_final 1: True
"""


# ---- contrastive: CORRECT vs WRONG trajectory (same TASK, contrasting reasoning) ----
FEW_SHOT_CONTRASTIVE_COT_CODE = """### CODE_COMPLETION FEW-SHOT EXAMPLES ###

=== FEW-SHOT EXAMPLE 1 ===
TASK:
Problem: from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"

Solution A:   for i in range(len(numbers) - 1):
    for j in range(i + 1, len(numbers)):
      if abs(numbers[i] - numbers[j]) > threshold:
        return False
  return True
Solution B:   return any(abs(a - b) < threshold for a, b in zip(numbers, numbers[1:]))
Solution C:   for i in range(len(numbers)):  # Change range to len(numbers)
    for j in range(i + 1, len(numbers)):
      if abs(numbers[i] - numbers[j]) < threshold:
        return True
  return False
Solution D:     for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False

[CORRECT TRAJECTORY]
Thought 1:
Return True iff some pair is closer than threshold. A returns False on the first FAR pair (inverted): WRONG. B only compares adjacent pairs via `zip(numbers, numbers[1:])`, missing non-neighbour close pairs: WRONG. C and D both do the full pairwise `< threshold` scan and are equivalent; the canonical reference is the enumerate double loop (D).
Action 1: db_search
Action Input 1: {'query': 'return true if any two elements closer than threshold pairwise enumerate'}
Is_final 1: False

Observation 1: query: return true if any two elements closer than threshold pairwise enumerate
retrieved context: [CHUNK 2101 | doc=0071]
for idx, elem in enumerate(numbers):
    for idx2, elem2 in enumerate(numbers):
        if idx != idx2 and abs(elem - elem2) < threshold:
            return True
return False

Thought 2:
The reference matches option D exactly. The answer is D.
Action 2: finish
Action Input 2: {}
Is_final 2: True

[WRONG TRAJECTORY]
Thought 1:
All four options look plausible. Option B is a neat one-liner with `any(...)`, and short idiomatic code is usually the intended answer, so I will go with the most concise version.
Action 1: finish
Action Input 1: {}
Is_final 1: True
# PITFALL: chose B for BREVITY. B compares only adjacent pairs (zip of numbers with
# numbers[1:]); it misses close pairs that are not neighbours, so it is a real bug.
# Concision is not a tie-break when an option changes behaviour. Correct answer: D.

=== FEW-SHOT EXAMPLE 2 ===
TASK:
Problem:
def generate_integers(a, b):
    \"\"\"
    Given two positive integers a and b, return the even digits between a
    and b, in ascending order.

    For example:
    generate_integers(2, 8) => [2, 4, 6, 8]
    generate_integers(8, 2) => [2, 4, 6, 8]
    generate_integers(10, 14) => []
    \"\"\"

Solution A:     lower = max(2, min(a, b))
    upper = min(8, max(a, b))

    return [i for i in range(lower, upper+1) if i % 2 != 0]
Solution B:     lower = max(2, min(a, b))
    upper = min(8, max(a, b))

    return [i for i in range(lower, upper+1) if i % 2 == 0]
Solution C:     lower = max(2, min(a, b))
    upper = min(8, max(a, b))

    return [i for i in range(lower, upper) if i % 2 == 0]
Solution D:     lower = min(2, min(a, b))
    upper = max(8, max(a, b))

    return [i for i in range(lower, upper+1) if i % 2 == 0]

[CORRECT TRAJECTORY]
Thought 1:
Even digits of [2,8] between a and b, ascending. A keeps ODD values (`i % 2 != 0`): WRONG. D fails to clamp to [2,8] (`min(2,...)`/`max(8,...)`): WRONG. C uses `range(lower, upper)`, dropping the inclusive upper bound 8 for (2,8): WRONG. B clamps to [2,8], iterates inclusively `range(lower, upper+1)`, keeps evens. The answer is B.
Action 1: finish
Action Input 1: {}
Is_final 1: True

[WRONG TRAJECTORY]
Thought 1:
Both B and C keep even numbers over the clamped [lower, upper]. `range(lower, upper)` reads naturally as "from lower up to upper", so C looks correct and a bit cleaner without the `+1`. I pick C.
Action 1: finish
Action Input 1: {}
Is_final 1: True
# PITFALL: `range(lower, upper)` is upper-EXCLUSIVE, so for (2, 8) it yields [2, 4, 6]
# and drops the required 8. The inclusive `range(lower, upper + 1)` (option B) is needed.
# Correct answer: B.

=== FEW-SHOT EXAMPLE 3 ===
TASK:
Problem:
def prime_length(string):
    \"\"\"Write a function that takes a string and returns True if the string
    length is a prime number or False otherwise
    Examples
    prime_length('Hello') == True
    prime_length('abcdcba') == True
    prime_length('kittens') == True
    prime_length('orange') == False
    \"\"\"

Solution A:     l = len(string)
    if l == 0 or l == 1:
        return False
    return l % 2 != 0
Solution B:     l = len(string)
    if l == 0 or l == 1:
        return False
    for i in range(2, l):
        if l % i == 0:
            return False
    return True
Solution C:     l = len(string)
    if l == 0 or l == 1:
        return False
    return l % 3 != 0
Solution D:     l = len(string)
    if l == 0 or l == 1:
        return False
    return l % 2 == 1

[CORRECT TRAJECTORY]
Thought 1:
Return True iff length is prime. A (`l % 2 != 0`) and D (`l % 2 == 1`) both only test ODDness, so length 9 returns True though 9 is not prime: WRONG. C (`l % 3 != 0`) is unrelated to primality: WRONG. B is a real trial-division primality test. The answer is B.
Action 1: finish
Action Input 1: {}
Is_final 1: True

[WRONG TRAJECTORY]
Thought 1:
The docstring examples have lengths 5, 7, 7, 6. For the True cases (5, 7) the length is odd; for the False case (6) it is even. So "prime length" seems to coincide with "odd length". Option A returns `l % 2 != 0`, which reproduces every listed example, so I pick A.
Action 1: finish
Action Input 1: {}
Is_final 1: True
# PITFALL: fitting only the DOCSTRING examples. Odd != prime: length 9 is odd but not
# prime, and A would wrongly return True. Reason on unseen edge cases, not just the given
# examples. The real primality test is option B. Correct answer: B.
"""


FEW_SHOT_CONTRASTIVE_COT_MIDDLE = """### FILL_IN_THE_MIDDLE FEW-SHOT EXAMPLES ###

=== FEW-SHOT EXAMPLE 1 ===
TASK:
Problem: class Solution:
  def dominantIndex(self, nums: List[int]) -> int:
    max = 0
    secondMax = 0

    for i, num in enumerate(nums):
      if num > max:
        secondMax = max
        max = num
        ans = i
      elif num > secondMax:
        secondMax = num

    _______________
    return ans if max >= 2 * secondMax else -1 You are given an integer array nums where the largest integer is unique. Determine whether the largest element in the array is at least twice as much as every other number in the array. If it is, return the index of the largest element, or return -1 otherwise.

Solution A:     return ans if max > 2 * secondMax else -1
Solution B:     return ans if max >= 2 * secondMax else -1
Solution C:     return ans if max == 2 * secondMax else -1
Solution D:     return ans if max // 2 >= secondMax else -1

[CORRECT TRAJECTORY]
Thought 1:
"At least twice" is `max >= 2 * secondMax`, which is also the line already shown after the blank. A (`>`) and C (`==`) drop or change the equality boundary: WRONG. D rewrites with integer division `max // 2 >= secondMax`, a non-canonical form. B is the direct canonical `>=`. The answer is B.
Action 1: finish
Action Input 1: {}
Is_final 1: True

[WRONG TRAJECTORY]
Thought 1:
"Twice as much" sounds like a strict multiple, so the largest must be more than twice the runner-up. `>` captures "more than twice", so I choose A.
Action 1: finish
Action Input 1: {}
Is_final 1: True
# PITFALL: misread "at LEAST twice" as "strictly more than twice". `>` (A) returns -1
# when max is exactly 2 * secondMax, which the spec accepts. The canonical `>=` is B.
# Correct answer: B.

=== FEW-SHOT EXAMPLE 2 ===
TASK:
Problem: class Solution:
  def makeLargestSpecial(self, s: str) -> str:
    specials = []
    count = 0

    i = 0
    for j, c in enumerate(s):
      count += 1 if c == '1' else -1
      if count == 0:
        specials.append(
            _______________
            '1' + self.makeLargestSpecial(s[i + 1:j]) + '0')
        i = j + 1

    return ''.join(sorted(specials)[::-1]) Special binary strings are binary strings with the following two properties: You are given a special binary string s. A move consists of choosing two consecutive, non-empty, special substrings of s, and swapping them. Two strings are consecutive if the last character of the first string is exactly one index before the first character of the second string. Return the lexicographically largest resulting string possible after applying the mentioned operations on the string.

Solution A:             self.makeLargestSpecial(s[i + 1:j]) + '10'
Solution B:             '10' + self.makeLargestSpecial(s[i + 1:j]) + '0'
Solution C:             '1' + self.makeLargestSpecial(s[i + 1:j]) + '01'
Solution D:             '1' + self.makeLargestSpecial(s[i + 1:j]) + '0')

[CORRECT TRAJECTORY]
Thought 1:
A balanced block is wrapped as `'1' + makeLargestSpecial(inner) + '0'`, and the fill must also close the `append(` call. The already-shown continuation is `'1' + self.makeLargestSpecial(s[i + 1:j]) + '0')`. A, B and C each mangle the `'1'...'0'` wrap. D reproduces the canonical wrap and closes the parenthesis. The answer is D.
Action 1: finish
Action Input 1: {}
Is_final 1: True

[WRONG TRAJECTORY]
Thought 1:
The blank is on its own line, so it is probably a standalone recursive term and the next line completes it. Option A, `self.makeLargestSpecial(s[i + 1:j]) + '10'`, is the shortest recursive-looking expression, so I choose A.
Action 1: finish
Action Input 1: {}
Is_final 1: True
# PITFALL: ignored that the fill must match the shown `'1' + ... + '0')` continuation and
# close the append() parenthesis. A drops the leading `'1'` and mis-wraps as `'10'`.
# Correct answer: D.

=== FEW-SHOT EXAMPLE 3 ===
TASK:
Problem: class Solution:
  def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
    while sx < tx and sy < ty:
      tx, ty = tx % ty, ty % tx

    _______________
    return sx == tx and sy <= ty and (ty - sy) % tx == 0 or \\
        sy == ty and sx <= tx and (tx - sx) % ty == 0 Given four integers sx, sy, tx, and ty, return true if it is possible to convert the point (sx, sy) to the point (tx, ty) through some operations, or false otherwise. The allowed operation on some point (x, y) is to convert it to either (x, x + y) or (x + y, y).

Solution A:     return sx == tx and sy < ty and (ty + sy) % tx == 0 or \\
Solution B:     return sx == tx and sy <= ty and (sx - sy) % tx == 0 or \\
Solution C:     return sx == tx and sy <= ty and (ty - sy) % tx == 0 or \\
Solution D:     return sx == tx and sy < ty and (ty - sy) % tx == 0 or \\

[CORRECT TRAJECTORY]
Thought 1:
The fill is the first line of the two-line return; it must mirror the shown second line as `sx == tx and sy <= ty and (ty - sy) % tx == 0`. A changes `<=`->`<` and `(ty - sy)`->`(ty + sy)`: WRONG. B uses `(sx - sy)`: WRONG. D changes `<=`->`<`: WRONG. C matches exactly. The answer is C.
Action 1: finish
Action Input 1: {}
Is_final 1: True

[WRONG TRAJECTORY]
Thought 1:
C and D both use `(ty - sy) % tx == 0`, which matches the mirrored line. Between `sy < ty` and `sy <= ty` the strict `<` looks safer against division-by-zero style edge cases, so I pick D.
Action 1: finish
Action Input 1: {}
Is_final 1: True
# PITFALL: swapped `<=` for `<`, dropping the boundary case sy == ty that the symmetric
# second line (`sx <= tx`) clearly keeps. The canonical mirror uses `<=` -> option C.
# Correct answer: C.
"""


# ---- auto_cot: compact reasoning trajectory ending with the chosen letter ----
FEW_SHOT_AUTO_COT_CODE = """### CODE_COMPLETION FEW-SHOT EXAMPLES ###
=== FEW-SHOT EXAMPLE 1===\nTASK:\nProblem: Implement a function that splits a given string into words based on spaces or commas, or counts the number of lowercase letters with even order in the alphabet if neither space nor comma is present.\n\nSolution A: Splits on spaces, then commas, otherwise counts lowercase letters with even order.\nSolution B: Splits on spaces, then commas, otherwise counts all letters with even order.\nSolution C: Splits on spaces, then commas, otherwise counts lowercase letters with odd order.\nSolution D: Splits on spaces, then commas, otherwise counts all lowercase letters.\n\nThought 1: Verify which solution correctly counts lowercase letters with even order in the alphabet when no spaces or commas are present.\nAction 1: llm\nAction Input 1: {'query': 'Compare solutions A, B, C, D for the case where no spaces or commas exist in the input.'}\nIs_final 1: False\n\nObservation 1: Solution A correctly implements the logic to count lowercase letters with even order in the alphabet when there are no spaces or commas in the input.\n\nThought 2: Confirm that the remaining solutions either have incorrect logic for counting letters or deviate from the specified behavior.\nAction 2: llm\nAction Input 2: {'query': 'Identify any deviations in solutions B, C, D from the correct behavior described in solution A.'}\nIs_final 2: False\n\nObservation 2: Solutions B, C, and D deviate from the required behavior by either counting all letters with even order (B), counting lowercase letters with odd order (C), or counting all lowercase letters (D).\n\nThought 3: Settle on the reference-correct option A as it adheres to the specified behavior.\nAction 3: finish\nAction Input 3: {}\nIs_final 3: True\nAnswer: A\nAnswer: Answer: A\n\n=== FEW-SHOT EXAMPLE 2===\nTASK:\nProblem: Given an integer, return a tuple indicating the count of even and odd digits.\n\nSolution A: Incorrectly increments even_count for odd numbers and vice versa.\nSolution B: Only counts even numbers, ignoring odd counts.\nSolution C: Incorrectly increments odd_count for even numbers and vice versa.\nSolution D: Correctly counts even and odd digits.\n\nThought 1: Need to check which solutions correctly count even and odd digits.\nAction 1: db_search\nAction Input 1: {'query': 'python count even and odd digits in integer'}\nIs_final 1: False\n\nObservation 1: query: python count even and odd digits in integer\nretrieved context: [CHUNK 12345 | doc=6789]\nIn [1]: def correct_even_odd_count(num):\n   ...:     even_count = 0\n   ...:     odd_count = 0\n   ...:     for i in str(abs(num)):\n   ...:         if int(i) % 2 == 0:\n   ...:             even_count += 1\n   ...:         else:\n   ...:             odd_count += 1\n   ...:     return (even_count, odd_count)\n   ...: \n   ...: print(correct_even_odd_count(-12))  # (1, 1)\n   ...: print(correct_even_odd_count(123))  # (1, 2)\n   ...: \n\nThought 2: Comparing the reference implementation with the given options, Solution D matches the reference-correct behavior.\nAction 2: finish\nAction Input 2: {}\nIs_final 2: True\nAnswer: D\nAnswer: Answer: D\n\n=== FEW-SHOT EXAMPLE 3===\nTASK:\nProblem: Define a function that takes a list and returns a tuple containing the largest negative integer and the smallest positive integer. Return None for missing values.\n\nSolution A: Filters negatives and non-negatives, returning their max/min, respectively.\nSolution B: Filters non-positive and positives, returning their max/min, respectively.\nSolution C: Filters negatives and positives, returning their min/max, respectively.\nSolution D: Filters negatives and positives, returning their max/min, respectively.\n\nThought 1: Check which solutions correctly filter and compute the max/min for negative and positive integers.\nAction 1: db_search\nAction Input 1: {'query': 'python filter max min on list'}\nIs_final 1: False\n\nObservation 1: \nquery: python filter max min on list\nretrieved context: [CHUNK 12345 | doc=6789]\nIn [1]: help(filter)\nOut[1]: Built-in functions\nfilter(function or None, iterable) --> filter object\nReturn an iterator yielding those items of iterable for which function(item) is true. If function is None, return the items that are true.\n\nThought 2: Compare options A through D based on their behavior in handling the filtering and calculation of max and min values for negative and positive integers.\nAction 2: llm\nAction Input 2: {'query': 'Compare Solution A, B, C, D for correct filtering and max/min computation for negative and positive integers.'}\nIs_final 2: False\n\nObservation 2: Solutions A and B incorrectly include zero in their filtering criteria, while Solution C returns the minimum negative and maximum positive integers instead of the maximum negative and minimum positive integers. Solution D correctly identifies the largest negative and smallest positive integers.\n\nThought 3: Confirm that Solution D is the only one that correctly implements the intended logic without changing the behavior.\nAction 3: finish\nAction Input 3: {}\nIs_final 3: True\nAnswer: D\nAnswer: Answer: D\n\n"""


FEW_SHOT_AUTO_COT_MIDDLE = """### FILL_IN_THE_MIDDLE FEW-SHOT EXAMPLES ###
=== FEW-SHOT EXAMPLE 1===\nTASK:\nProblem: Given a tree structure defined by parent-child relationships and genetic values for each node, find the smallest missing genetic value for each subtree rooted at each node.\n\nSolution A: Initializes the answer array with 1s.\nSolution B: Initializes the answer array with -1s.\nSolution C: Initializes the answer array with infinity.\nSolution D: Initializes the answer array with 0s.\n\nThought 1: The correct initialization should ensure that the initial smallest missing genetic value is 1, as the genetic values range from 1 to 10^5.\nAction 1: db_search\nAction Input 1: {'query': 'smallestMissingValueSubtree problem canonical solution'}\nIs_final 1: False\n\nObservation 1: query: smallestMissingValueSubtree problem canonical solution\nretrieved context: [CHUNK 56789 | doc=12345]\nIn [1]: ans = [1] * n\nOut[1]: [1, 1, 1, ..., 1]  # n times\n\nThought 2: Option A correctly initializes the answer array with 1s, which aligns with the canonical solution and ensures the correct starting point for finding the smallest missing genetic value.\nAction 2: finish\nAction Input 2: {}\nIs_final 2: True\nAnswer: A\nAnswer: Answer: A\n\n=== FEW-SHOT EXAMPLE 2===\nTASK:\nProblem: Given an integer array, calculate the sum of beauty values for elements between indices 1 and nums.length - 2 based on specific conditions.\n\nSolution A: Initializes minOfRight with zeros of length n.\nSolution B: Initializes minOfRight with zeros of length n-1 and the last element of nums.\nSolution C: Initializes minOfRight with the last element of nums repeated n times.\nSolution D: Initializes minOfRight with the first element of nums repeated n-1 times and the last element of nums.\n\nThought 1: Determine which initialization of minOfRight correctly aligns with the logic needed for calculating the sum of beauties.\nAction 1: db_search\nAction Input 1: {'query': 'initialization of minOfRight in sumOfBeauties'}\nIs_final 1: False\n\nObservation 1: query: initialization of minOfRight in sumOfBeauties\nretrieved context: [CHUNK 12345 | doc=6789]\nIn [1]: minOfRight = [0] * (n - 1) + [nums[-1]]\n\nThought 2: Option B correctly initializes minOfRight with zeros of length n-1 followed by the last element of nums, aligning with the reference implementation.\nAction 2: llm\nAction Input 2: {'query': 'compare solutions A, B, C, D based on correctness of minOfRight initialization'}\nIs_final 2: False\n\nObservation 2: Solutions A, C, and D do not initialize minOfRight correctly according to the problem's requirements. Only B follows the correct pattern.\n\nThought 3: Confirming B as the correct solution due to its accurate initialization of minOfRight.\nAction 3: finish\nAction Input 3: {}\nIs_final 3: True\nAnswer: B\nAnswer: Answer: B\n\n=== FEW-SHOT EXAMPLE 3===\nTASK:\nProblem: Calculate the total appeal of all substrings of a given string, where appeal is defined as the number of distinct characters in each substring.\n\nSolution A: Subtracts the difference between the current index and the last seen index from dp.\nSolution B: Adds the sum of the current index and the last seen index to dp.\nSolution C: Adds the difference between the last seen index and the current index to dp.\nSolution D: Adds the difference between the current index and the last seen index to dp.\n\nThought 1: To determine the correct solution, we need to understand how the appeal of substrings is calculated and which operation aligns with this calculation.\nAction 1: db_search\nAction Input 1: {'query': 'appeal of substrings calculation'}\nIs_final 1: False\n\nObservation 1: query: appeal of substrings calculation\nretrieved context: [CHUNK 12345 | doc=6789]\nIn [1]: The appeal of a substring is the number of distinct characters in it. For calculating the total appeal, one approach is to maintain a running total (dp) that increments based on the current character's index and its last seen position.\n\nThought 2: Based on the observation, the correct approach involves adding the difference between the current index and the last seen index of the character to dp. This ensures the total appeal reflects the correct count of distinct characters in all substrings.\nAction 2: llm\nAction Input 2: {'query': 'Compare solutions A, B, C, D for correctness.'}\nIs_final 2: False\n\nObservation 2: Comparing the solutions, only D correctly implements the logic for maintaining a running total that reflects the appeal of all substrings by considering the current character's position and its last seen position.\n\nThought 3: After verifying the logic, it is clear that Solution D correctly calculates the appeal of all substrings.\nAction 3: finish\nAction Input 3: {}\nIs_final 3: True\nAnswer: D\nAnswer: Answer: D\n\n"""


# Trailing marker appended after the examples, right before the live task is concatenated
# (mirrors the DS-1000 registry, where each few-shot string ends with "=== TASK ===").
_TASK_MARKER = "\n=== TASK ===\n"

# Keys mirror the DS-1000 registry so `few_shot_type` validation in the pipeline keeps
# working. Because one pipeline instance runs BOTH sub-splits (code, then middle) with a
# single few-shot string, the base keys carry the code AND middle examples. The
# `*_code` / `*_middle` keys expose each sub-type on its own, in case the pipeline is
# later changed to pick few-shots per sub-type.
FEW_SHOT_REGISTRY = {
    "zero_shot": "",
    "cot": FEW_SHOT_COT_CODE + "\n" + FEW_SHOT_COT_MIDDLE + _TASK_MARKER,
    "contrastive_cot": FEW_SHOT_CONTRASTIVE_COT_CODE + "\n" + FEW_SHOT_CONTRASTIVE_COT_MIDDLE + _TASK_MARKER,
    "auto_cot": FEW_SHOT_AUTO_COT_CODE + "\n" + FEW_SHOT_AUTO_COT_MIDDLE + _TASK_MARKER,
    "cot_code": FEW_SHOT_COT_CODE + _TASK_MARKER,
    "cot_middle": FEW_SHOT_COT_MIDDLE + _TASK_MARKER,
    "contrastive_cot_code": FEW_SHOT_CONTRASTIVE_COT_CODE + _TASK_MARKER,
    "contrastive_cot_middle": FEW_SHOT_CONTRASTIVE_COT_MIDDLE + _TASK_MARKER,
    "auto_cot_code": FEW_SHOT_AUTO_COT_CODE + _TASK_MARKER,
    "auto_cot_middle": FEW_SHOT_AUTO_COT_MIDDLE + _TASK_MARKER,
}
