"""Prompt templates for the ReWOO-SGR pipeline (structured-output Planner/Worker/Solver)
— CodeMMLU dataset.

CodeMMLU is a multiple-choice benchmark: each task gives a Python code problem and
four candidate options (A, B, C, D); exactly one is the reference-correct answer.
Two sub-types are covered:
  * code_completion   — a function (docstring + examples) and four completions;
                        failures are mostly REAL bugs -> reason on edge cases.
  * fill_in_the_middle — a fragment with a missing line and four options that are
                        often FUNCTIONALLY EQUIVALENT -> pick the canonical form.

Mirrors the DS-1000 file at `prompts.ds1000.prompts_rewoo_sgr` and exposes the SAME
public names (PLANNER_PROMPT, WORKER_PROMPT, SOLVER_PROMPT, FEW_SHOT_REGISTRY), so the
pipeline can switch datasets by changing only the import path. Imported back via
`from src.agents.pipelines.prompts.codemmlu.prompts_rewoo_sgr import ...`.

Few-shot examples come in two families, matching how the pipeline consumes them
(see rewoo_sgr.py): the PLANNER few-shots (`cot`, `contrastive_cot`, `auto_cot`) are
prepended to PLANNER_PROMPT and demonstrate a structured JSON plan; the SOLVER
few-shots (`solver_cot`, `solver_contrastive_cot`) are prepended to SOLVER_PROMPT and
demonstrate the final `{"thought": ..., "response": "<letter>"}` structured output.
Each family is provided separately for the two sub-types (code / middle) and combined
in the registry, exactly like the sibling `prompts_react_sgr` file.

Every TASK below is produced the way `build_agent_task()` assembles a live task
(see codemmlu_agent_pipelines.py): a "Problem: {input}" line followed by the four
"Solution A/B/C/D" options. The only difference between the two sub-types is `input`:
  * code_completion   -> input = question
  * fill_in_the_middle -> input = question + " " + problem_description
The TASKs are taken verbatim from examples.json (3 code_completion + 3 fill_in_the_middle).

NOTE on braces: PLANNER_PROMPT and SOLVER_PROMPT are passed through str.format(), so any
literal brace must be doubled ({{ }}). The FEW_SHOT_* strings are NOT formatted — they are
concatenated as-is — so their JSON braces stay single.
"""

PLANNER_PROMPT = """For the following MULTIPLE-CHOICE code question, make a plan that decides which single \
option — A, B, C, or D — is the reference-correct answer. You are given a code problem and four \
candidate options; EXACTLY ONE is reference-correct. For each plan step, indicate which external \
tool together with tool input to retrieve evidence. You can store the evidence into a variable #E \
that can be called by later tools.

TOOLS (Case-Sensitive, Exact Names Only)
{tools_formatted}

You must respond with a structured JSON format containing a list of steps. Each step should have:
- step_id: integer (starting from 1)
- plan: string description of what this step does
- tool: string name of the tool to use
- args: object with tool-specific arguments
- evidence_tag: string like "#E1", "#E2", etc.
- depends_on: list of evidence tags this step depends on (e.g., ["#E1"] or [])

HOW TO DECIDE THE LETTER (drive the plan toward this)
STEP 1 — Reject any option that changes BEHAVIOUR. Plan to mentally run each option on the cases the
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
`return a, b` (not `return [a, b]`); `1 + max(...)` (not `max(...) + 1`); strict `>` to track a
maximum, keeping the FIRST among equal values; no extra guard/cast/paren the spec does not need;
PEP 8 spacing `[1] * (n + 1)` (not `[1]*(n+1)`). Judge by the FORM, not by the position — the
canonical option is frequently A; never reject or avoid an option just because of where it appears.

Use the tools to gather evidence (e.g. retrieve the canonical reference form for the described
routine, or confirm a boundary behaviour) that supports settling on ONE letter. Keep the plan
minimal — one or two well-targeted steps are usually enough; do not add redundant duplicate searches.

CONSTRAINTS
1. Action MUST be one of: {tool_names} or 'finish'.
   IMPORTANT: Few-shot examples may demonstrate different tools for format/structure reference only. \
You must strictly call ONLY the available tools listed in {tool_names}.
2. args MUST match the tool's argument schema exactly (a JSON object, e.g. {{"query": "..."}}), \
never a raw string.

For example:
Task: A multiple-choice question asks which of four completions correctly returns True iff any two
numbers in a list are closer than a threshold.

{{
  "steps": [
    {{
      "step_id": 1,
      "plan": "Retrieve the canonical reference implementation for 'return True if any two elements are closer than threshold' to compare each option against.",
      "tool": "db_search",
      "args": {{"query": "return true if any two elements are closer than threshold pairwise nested loop enumerate"}},
      "evidence_tag": "#E1",
      "depends_on": []
    }},
    {{
      "step_id": 2,
      "plan": "Given the retrieved reference #E1, decide which option reproduces its behaviour on the docstring examples and boundaries, rejecting inverted-logic and adjacent-only variants.",
      "tool": "llm",
      "args": {{"query": "Given reference #E1, which of options A/B/C/D matches its behaviour exactly?"}},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }}
  ]
}}

Begin!
Describe your plan with rich details.

Task: {task}"""


WORKER_PROMPT = """Respond in short directly with no extra words.\n\n{request}"""


SOLVER_PROMPT = """You are providing the FINAL ANSWER to a MULTIPLE-CHOICE code question. To decide, we have \
made a step-by-step Plan and retrieved corresponding Evidence for each Plan. Use them with caution \
since long evidence might contain irrelevant information.

{plan}

INSTRUCTIONS:
1. The answer is a SINGLE letter — A, B, C, or D. There is exactly one reference-correct option.
2. Reject any option that changes BEHAVIOUR (wrong comparison/operator/variable, off-by-one, wrong
   precedence, a dropped guard the spec needs, a wrong formula or algorithm). Keep the option whose
   result matches EVERY example exactly.
3. TIE-BREAK: If the surviving options are functionally equivalent, keep the canonical / minimal
   form (the simplest standard idiom), judging by FORM not position — the canonical option is
   frequently A; never avoid an option just because of where it appears.
4. Use the Evidence only when it gives a concrete, verifiable reason; ignore evidence about style,
   naming, or theoretical edge cases that do not change which option is reference-correct.

OUTPUT RULES:
- In the "response" field output EXACTLY ONE character — A, B, C, or D. No words, no code, no
  punctuation, no explanation.

Task: {task}
Response:"""


# =====================================================================================
# PLANNER FEW-SHOT EXAMPLES
# =====================================================================================
# Prepended to PLANNER_PROMPT (see rewoo_sgr.py PlannerREWOOSGR.run). Each example is a
# TASK + a structured JSON plan; the tools shown (db_search / llm) are for FORMAT reference
# only — at run time the planner must call ONLY the registered tools (see PLANNER_PROMPT
# CONSTRAINTS). The `cot` / `auto_cot` variants add a THOUGHT before the plan; the
# `contrastive_cot` variant contrasts a CORRECT plan with a WRONG one.


# ---- cot: THOUGHT then a clean structured plan (code_completion) --------------------
FEW_SHOT_COT_CODE = """### CODE_COMPLETION FEW-SHOT EXAMPLES (PLANNER) ###

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

THOUGHT:
The function must return True iff SOME pair is closer than `threshold`. A returns False on the first
FAR pair (`> threshold`) — inverted logic, WRONG. B compares only `zip(numbers, numbers[1:])`, i.e.
ADJACENT pairs, missing non-neighbour close pairs, WRONG. C and D both scan every pair and return
True on `< threshold`; they are functionally EQUIVALENT, so this is a genuine tie and I keep the
canonical full pairwise enumerate form (D). I will retrieve the canonical reference to confirm the
tie-break, then decide the letter.

BEGIN PLAN
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical reference implementation for 'return True if any two elements are closer than threshold' to compare each option against.",
      "tool": "db_search",
      "args": {"query": "return true if any two elements are closer than threshold pairwise nested loop enumerate"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1, decide which option reproduces its behaviour on the docstring examples and boundaries, rejecting the inverted (A) and adjacent-only (B) variants and, between the equivalent C and D, keeping the canonical enumerate double loop.",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which of options A/B/C/D matches its behaviour exactly, preferring the canonical full pairwise enumerate form?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}

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

THOUGHT:
The spec wants the EVEN one-digit values in [2, 8] between a and b, ascending. A keeps `i % 2 != 0`
(odd values), WRONG. C iterates `range(lower, upper)`, dropping the inclusive upper bound 8, WRONG.
D clamps with `min(2, ...)`/`max(8, ...)`, which fails to bound the interval to [2, 8], WRONG. B
clamps to [2, 8], iterates inclusively, and keeps evens. I will retrieve the inclusive-clamp
reference to confirm the boundary, then decide.

BEGIN PLAN
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical reference for clamping to [2, 8] and keeping even numbers over an inclusive range, to confirm the boundary handling.",
      "tool": "db_search",
      "args": {"query": "inclusive range clamp lower upper keep even numbers list comprehension max min"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1, decide which option clamps to [2, 8], iterates inclusively `range(lower, upper+1)`, and keeps even values, rejecting the odd-filter (A), the exclusive-range off-by-one (C), and the unclamped bounds (D).",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which of options A/B/C/D clamps to [2,8], iterates inclusively, and keeps even numbers?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}

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

THOUGHT:
The function must return True iff `len(string)` is prime. A (`l % 2 != 0`) and D (`l % 2 == 1`) are
the SAME odd test — length 9 is odd but not prime, so both are WRONG. C (`l % 3 != 0`) is unrelated
to primality (length 25 -> True though 25 is not prime), WRONG. B is a real trial-division primality
test over `range(2, l)`. I will retrieve the canonical trial-division form to confirm, then decide.

BEGIN PLAN
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical primality-by-trial-division reference to confirm which option is a genuine primality test rather than an odd/mod shortcut.",
      "tool": "db_search",
      "args": {"query": "primality test trial division for i in range 2 to n return false if divisible"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1, decide which option performs true trial division, rejecting the odd-only checks (A, D) and the mod-3 shortcut (C).",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which of options A/B/C/D is a real trial-division primality test?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}
"""


# ---- cot: THOUGHT then a clean structured plan (fill_in_the_middle) -----------------
FEW_SHOT_COT_MIDDLE = """### FILL_IN_THE_MIDDLE FEW-SHOT EXAMPLES (PLANNER) ###

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

THOUGHT:
The spec says "at least twice as much", i.e. `max >= 2 * secondMax`; the reference line already shown
after the blank is exactly that, so the fill must reproduce the canonical `>=`. A uses `>` (drops the
equality boundary), C uses `==` (only the exact-double case), D rewrites as `max // 2 >= secondMax`
(non-canonical integer division). B is the direct canonical form. I will retrieve the canonical
"at least twice" comparison to confirm, then decide.

BEGIN PLAN
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical form for the phrase 'x is at least twice y' to confirm it maps to `x >= 2 * y` and keeps the equality boundary.",
      "tool": "db_search",
      "args": {"query": "at least twice as large comparison canonical greater than or equal 2 times keep equality"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1 and the already-shown `max >= 2 * secondMax` line, decide which option reproduces the canonical `>=`, rejecting `>` (A), `==` (C), and the integer-division rewrite (D).",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which option matches the canonical `max >= 2 * secondMax` comparison shown after the blank?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}

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

THOUGHT:
Each balanced block is rebuilt as `'1' + makeLargestSpecial(inner) + '0'`. The continuation already
shown after the blank is `'1' + self.makeLargestSpecial(s[i + 1:j]) + '0')`, so the fill must
reproduce that exact `'1'...'0'` wrap and close the `append(` call. A drops the leading `'1'` and
mangles it to `'10'`; B prepends `'10'`; C appends `'01'` instead of `'0'`. D is the canonical wrap
that also closes the parenthesis. I will confirm the wrapping convention, then decide.

BEGIN PLAN
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical recursive wrap for makeLargestSpecial to confirm a block is rebuilt as `'1' + rec + '0'`.",
      "tool": "db_search",
      "args": {"query": "makeLargestSpecial special binary string wrap '1' + recurse + '0' append"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1 and the already-shown `'1' + ... + '0')` continuation, decide which option reproduces the `'1'...'0'` wrap and closes the append() parenthesis, rejecting the mangled wraps (A, B, C).",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which option matches the `'1' + self.makeLargestSpecial(s[i + 1:j]) + '0'` wrap and closes the append paren?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}

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

THOUGHT:
The blank is the FIRST physical line of the two-line return; the second line
`sy == ty and sx <= tx and (tx - sx) % ty == 0` is already shown, so the fill must be its symmetric
counterpart `sx == tx and sy <= ty and (ty - sy) % tx == 0`. A uses `sy < ty` and `(ty + sy)` (wrong
boundary + operator); B uses `(sx - sy)` (wrong operands); D uses `sy < ty` (drops the equality
boundary). C matches the canonical mirror exactly. I will confirm the symmetry, then decide.

BEGIN PLAN
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical symmetric reachingPoints return to confirm the mirrored first line uses `sy <= ty` and `(ty - sy) % tx == 0`.",
      "tool": "db_search",
      "args": {"query": "reachingPoints return symmetric sx == tx sy <= ty (ty - sy) % tx == 0 mirror"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1 and the already-shown second line, decide which option is its exact symmetric mirror, rejecting the `<` boundary change (A, D) and the wrong operands `(ty + sy)`/`(sx - sy)` (A, B).",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which option mirrors the shown second line as `sx == tx and sy <= ty and (ty - sy) % tx == 0`?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}
"""


# ---- contrastive_cot: CORRECT plan vs WRONG plan (same TASK) ------------------------
FEW_SHOT_CONTRASTIVE_COT_CODE = """### CODE_COMPLETION FEW-SHOT EXAMPLES (PLANNER, CONTRASTIVE) ###

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
THOUGHT:
A is inverted (returns False on a far pair); B compares only adjacent pairs; C and D are equivalent
full pairwise scans, and D is the canonical enumerate form. A clean two-step plan retrieves the
reference and then settles the letter, passing args as proper JSON objects.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical reference implementation for 'return True if any two elements are closer than threshold'.",
      "tool": "db_search",
      "args": {"query": "return true if any two elements are closer than threshold pairwise nested loop enumerate"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1, decide which option matches it, rejecting inverted (A) and adjacent-only (B) and keeping the canonical enumerate form between the equivalent C and D.",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which of A/B/C/D matches the full pairwise behaviour, preferring the canonical enumerate double loop?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}

[WRONG TRAJECTORY]
THOUGHT:
This plan picks the shortest option for brevity and pads with redundant duplicate searches instead
of a targeted reference lookup. Option B is a neat one-liner, so it biases the decision toward B —
but B only compares ADJACENT pairs and is a real bug. Concision is not a tie-break when an option
changes behaviour, and re-running the same search wastes calls without new evidence.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Search for the shortest one-liner solution because concise idiomatic code is usually the intended answer.",
      "tool": "db_search",
      "args": {"query": "shortest one-liner any abs difference threshold zip"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Search the same idea again to be extra sure the concise option is correct.",
      "tool": "db_search",
      "args": {"query": "shortest one-liner any abs difference threshold zip"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}

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
THOUGHT:
A keeps odds; C is upper-exclusive and drops 8; D fails to clamp to [2, 8]; B clamps inclusively and
keeps evens. The plan retrieves the inclusive-clamp reference, then settles the letter.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical inclusive-clamp reference (clamp to [2,8], keep even numbers over range(lower, upper+1)).",
      "tool": "db_search",
      "args": {"query": "inclusive range clamp lower upper keep even numbers list comprehension max min"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1, decide which option clamps to [2,8], iterates inclusively, and keeps evens — rejecting the odd filter (A), exclusive range (C), and unclamped bounds (D).",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which of A/B/C/D clamps to [2,8], is inclusive, and keeps even numbers?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}

[WRONG TRAJECTORY]
THOUGHT:
This plan reads `range(lower, upper)` as "from lower up to upper" and treats C as cleaner without the
`+1`, biasing toward C. But `range(lower, upper)` is upper-EXCLUSIVE, so for (2, 8) it yields
[2, 4, 6] and drops the required 8. Preferring the shorter form over the inclusive boundary is the
error. It also passes `args` as a raw string instead of a JSON object.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Search for the cleaner exclusive-range version without the +1 since it reads more naturally.",
      "tool": "db_search",
      "args": "even numbers range lower upper clamp exclusive",
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}

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
THOUGHT:
A and D only test oddness (length 9 is odd but not prime); C tests divisibility by 3, unrelated to
primality; B is a genuine trial-division primality test. The plan retrieves the trial-division
reference, then settles on B.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical trial-division primality reference.",
      "tool": "db_search",
      "args": {"query": "primality test trial division for i in range 2 to n return false if divisible"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1, decide which option is a real primality test, rejecting the odd-only shortcuts (A, D) and the mod-3 shortcut (C).",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which of A/B/C/D is a genuine trial-division primality test?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}

[WRONG TRAJECTORY]
THOUGHT:
This plan fits only the DOCSTRING lengths (5, 7, 7 -> True are odd; 6 -> False is even) and concludes
"prime length" coincides with "odd length", biasing toward A. That is fitting the given examples
instead of reasoning on unseen edge cases: length 9 is odd but not prime, and A would wrongly return
True. It also finishes immediately with no reference lookup.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Conclude that the answer is the odd-length check because it fits every docstring example, and finish without verification.",
      "tool": "finish",
      "args": {},
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}
"""


FEW_SHOT_CONTRASTIVE_COT_MIDDLE = """### FILL_IN_THE_MIDDLE FEW-SHOT EXAMPLES (PLANNER, CONTRASTIVE) ###

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
THOUGHT:
"At least twice" is `max >= 2 * secondMax`, also the line already shown after the blank. A (`>`) and
C (`==`) change the equality boundary; D uses non-canonical integer division. B is the direct
canonical form. The plan confirms the canonical comparison and settles on B.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical form for 'x is at least twice y' to confirm it maps to `x >= 2 * y`.",
      "tool": "db_search",
      "args": {"query": "at least twice as large canonical greater than or equal 2 times keep equality"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1 and the already-shown `max >= 2 * secondMax`, decide which option reproduces the canonical `>=`, rejecting `>` (A), `==` (C), and integer division (D).",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which option matches the canonical `max >= 2 * secondMax` line?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}

[WRONG TRAJECTORY]
THOUGHT:
This plan misreads "at LEAST twice" as "strictly more than twice", so it biases toward `>` (A). But
`>` returns -1 when max is exactly `2 * secondMax`, a case the spec accepts, and it also contradicts
the `>=` line already shown after the blank. Reasoning from a misread phrase instead of the shown
reference is the error.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Assume 'twice as much' means strictly more than twice and pick the `>` comparison, finishing without checking the shown reference line.",
      "tool": "finish",
      "args": {},
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}

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
THOUGHT:
A block is wrapped as `'1' + makeLargestSpecial(inner) + '0'`, and the fill must also close the
`append(` call. The already-shown continuation is `'1' + self.makeLargestSpecial(s[i + 1:j]) + '0')`.
A, B and C each mangle the `'1'...'0'` wrap; D reproduces it and closes the parenthesis. The plan
confirms the wrap and settles on D.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical recursive wrap `'1' + rec + '0'` for makeLargestSpecial.",
      "tool": "db_search",
      "args": {"query": "makeLargestSpecial wrap '1' + recurse + '0' append special binary string"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1 and the shown `'1' + ... + '0')` continuation, decide which option matches the wrap and closes the append paren, rejecting the mangled wraps (A, B, C).",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which option matches `'1' + self.makeLargestSpecial(s[i + 1:j]) + '0'` and closes the append paren?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}

[WRONG TRAJECTORY]
THOUGHT:
This plan treats the blank as a standalone recursive term and picks the shortest recursive-looking
expression (A) for brevity. It ignores that the fill must match the shown `'1' + ... + '0')`
continuation and close the append() parenthesis — A drops the leading `'1'` and mis-wraps as `'10'`.
Choosing by length over the shown structure is the error.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Pick the shortest recursive-looking expression as the fill and finish without matching the shown continuation.",
      "tool": "finish",
      "args": {},
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}

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
THOUGHT:
The fill is the first line of the two-line return and must mirror the shown second line as
`sx == tx and sy <= ty and (ty - sy) % tx == 0`. A changes `<=`->`<` and `-`->`+`; B uses
`(sx - sy)`; D changes `<=`->`<`. C matches exactly. The plan confirms the symmetry and settles on C.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical symmetric reachingPoints return to confirm the mirrored first line uses `sy <= ty` and `(ty - sy) % tx == 0`.",
      "tool": "db_search",
      "args": {"query": "reachingPoints return symmetric sx == tx sy <= ty (ty - sy) % tx == 0 mirror"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "Given reference #E1 and the shown second line, decide which option is its exact symmetric mirror, rejecting the `<` boundary (A, D) and the wrong operands (A, B).",
      "tool": "llm",
      "args": {"query": "Given reference #E1, which option mirrors `sx == tx and sy <= ty and (ty - sy) % tx == 0`?"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}

[WRONG TRAJECTORY]
THOUGHT:
This plan notes both C and D use `(ty - sy) % tx == 0` and then prefers the strict `<` (D) as "safer
against division-by-zero edge cases", biasing toward D. That drops the boundary case `sy == ty` that
the symmetric second line (`sx <= tx`) clearly keeps. Inventing a safety rationale over the shown
mirror is the error.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Prefer the strict `sy < ty` version as safer and finish without checking the symmetry of the shown second line.",
      "tool": "finish",
      "args": {},
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}
"""


# ---- auto_cot: compact THOUGHT + plan, ending with the chosen letter ----------------
FEW_SHOT_AUTO_COT_CODE = """### CODE_COMPLETION FEW-SHOT EXAMPLES (PLANNER, AUTO-COT) ###

=== FEW-SHOT EXAMPLE 1 ===
TASK:
Problem: def has_close_elements(numbers, threshold): return True if any two numbers are closer than `threshold`, else False. Examples: [1.0,2.0,3.0],0.5 -> False; [1.0,2.8,3.0,4.0,5.0,2.0],0.3 -> True.

Solution A: nested loop, returns False when a pair is `> threshold` (inverted).
Solution B: `any(abs(a-b) < threshold for a,b in zip(numbers, numbers[1:]))` (adjacent pairs only).
Solution C: nested loop over all pairs, returns True when `< threshold`, else False.
Solution D: nested enumerate over all pairs with `idx != idx2`, returns True when `< threshold`, else False.

THOUGHT: A is inverted; B checks only adjacent pairs; C and D are equivalent full pairwise scans and D is the canonical enumerate form. Retrieve the reference, then decide.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the canonical pairwise reference and pick the matching canonical option.",
      "tool": "db_search",
      "args": {"query": "return true if any two elements closer than threshold pairwise enumerate"},
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}
Answer: D

=== FEW-SHOT EXAMPLE 2 ===
TASK:
Problem: def generate_integers(a, b): return the even digits in [2, 8] that lie between a and b, ascending. generate_integers(2,8) -> [2,4,6,8]; generate_integers(10,14) -> [].

Solution A: clamp [2,8] inclusive, keep `i % 2 != 0` (odd).
Solution B: clamp [2,8] inclusive `range(lower, upper+1)`, keep `i % 2 == 0`.
Solution C: clamp [2,8] but `range(lower, upper)` (upper-exclusive), keep evens.
Solution D: bounds `min(2,...)`/`max(8,...)` (not clamped), inclusive, keep evens.

THOUGHT: A returns odds; C drops the inclusive upper bound; D fails to clamp to [2,8]; B clamps correctly and iterates inclusively over evens. Retrieve the inclusive-clamp reference, then decide.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the inclusive-clamp reference (clamp [2,8], keep evens over range(lower, upper+1)) and pick the matching option.",
      "tool": "db_search",
      "args": {"query": "inclusive range clamp keep even numbers list comprehension max min"},
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}
Answer: B

=== FEW-SHOT EXAMPLE 3 ===
TASK:
Problem: def prime_length(string): return True iff len(string) is prime. 'Hello' -> True (5), 'orange' -> False (6).

Solution A: `return l % 2 != 0` (odd check).
Solution B: trial division `for i in range(2, l): if l % i == 0: return False; return True`.
Solution C: `return l % 3 != 0`.
Solution D: `return l % 2 == 1` (odd check).

THOUGHT: A and D only test oddness (length 9 is odd but not prime); C tests divisibility by 3; B is a genuine trial-division primality test. Retrieve the trial-division reference, then decide.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Retrieve the trial-division primality reference and pick the genuine primality test.",
      "tool": "db_search",
      "args": {"query": "primality test trial division for i in range 2 to n return false if divisible"},
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}
Answer: B
"""


FEW_SHOT_AUTO_COT_MIDDLE = """### FILL_IN_THE_MIDDLE FEW-SHOT EXAMPLES (PLANNER, AUTO-COT) ###

=== FEW-SHOT EXAMPLE 1 ===
TASK:
Problem: dominantIndex fills the blank before `return ans if max >= 2 * secondMax else -1`; the spec wants "at least twice as much".

Solution A: `return ans if max > 2 * secondMax else -1`
Solution B: `return ans if max >= 2 * secondMax else -1`
Solution C: `return ans if max == 2 * secondMax else -1`
Solution D: `return ans if max // 2 >= secondMax else -1`

THOUGHT: "At least twice" is `max >= 2 * secondMax`; A (`>`) and C (`==`) change the equality boundary; D uses non-canonical integer division. B matches the shown line. Confirm the canonical form, then decide.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Confirm the canonical 'at least twice' comparison `x >= 2 * y` and pick the matching option.",
      "tool": "db_search",
      "args": {"query": "at least twice canonical greater than or equal 2 times keep equality"},
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}
Answer: B

=== FEW-SHOT EXAMPLE 2 ===
TASK:
Problem: makeLargestSpecial fills the blank inside `specials.append(...)`; the shown continuation is `'1' + self.makeLargestSpecial(s[i + 1:j]) + '0')`, so a block is wrapped as `'1' + rec + '0'`.

Solution A: `self.makeLargestSpecial(s[i + 1:j]) + '10'`
Solution B: `'10' + self.makeLargestSpecial(s[i + 1:j]) + '0'`
Solution C: `'1' + self.makeLargestSpecial(s[i + 1:j]) + '01'`
Solution D: `'1' + self.makeLargestSpecial(s[i + 1:j]) + '0')`

THOUGHT: The fill must reproduce the `'1' + rec + '0'` wrap and close the append() paren. A, B and C each mangle the wrap; D matches and closes the parenthesis. Confirm the wrap, then decide.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Confirm the canonical `'1' + rec + '0'` wrap and pick the option that closes the append paren.",
      "tool": "db_search",
      "args": {"query": "makeLargestSpecial wrap '1' + recurse + '0' append"},
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}
Answer: D

=== FEW-SHOT EXAMPLE 3 ===
TASK:
Problem: reachingPoints fills the first line of a two-line return; the shown second line is `sy == ty and sx <= tx and (tx - sx) % ty == 0`, so the fill is its mirror.

Solution A: `sx == tx and sy < ty and (ty + sy) % tx == 0 or \\`
Solution B: `sx == tx and sy <= ty and (sx - sy) % tx == 0 or \\`
Solution C: `sx == tx and sy <= ty and (ty - sy) % tx == 0 or \\`
Solution D: `sx == tx and sy < ty and (ty - sy) % tx == 0 or \\`

THOUGHT: The canonical mirror is `sx == tx and sy <= ty and (ty - sy) % tx == 0`. A changes `<=`->`<` and `-`->`+`; B uses `(sx - sy)`; D changes `<=`->`<`. C matches exactly. Confirm the symmetry, then decide.
{
  "steps": [
    {
      "step_id": 1,
      "plan": "Confirm the symmetric mirror `sx == tx and sy <= ty and (ty - sy) % tx == 0` and pick the matching option.",
      "tool": "db_search",
      "args": {"query": "reachingPoints symmetric return sy <= ty (ty - sy) % tx == 0 mirror"},
      "evidence_tag": "#E1",
      "depends_on": []
    }
  ]
}
Answer: C
"""


# =====================================================================================
# SOLVER FEW-SHOT EXAMPLES
# =====================================================================================
# Prepended to SOLVER_PROMPT (see rewoo_sgr.py SolverREWOOSGR.run). The Plan/Evidence
# bullets are shaped exactly like `_build_completed_plan_str` produces them, and each
# example ends with the structured output the solver must emit: a "thought" plus a
# "response" that is EXACTLY ONE letter (A/B/C/D).


# ---- solver_cot: Plan + Evidence -> {thought, response:<letter>} (code_completion) --
SOLVER_FEW_SHOT_COT_CODE = """### CODE_COMPLETION FEW-SHOT EXAMPLES (SOLVER) ###

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

- Plan: 'Retrieve the canonical reference implementation for returning True if any two elements are closer than threshold.'
- Evidence: 'def has_close_elements(numbers, threshold):\\n    for idx, elem in enumerate(numbers):\\n        for idx2, elem2 in enumerate(numbers):\\n            if idx != idx2 and abs(elem - elem2) < threshold:\\n                return True\\n    return False'
- Plan: 'Decide which option reproduces the reference behaviour, rejecting inverted and adjacent-only variants.'
- Evidence: 'Option A returns False on the first FAR pair (inverted). Option B compares only adjacent pairs via zip. Options C and D are equivalent full pairwise scans; D is the canonical enumerate form matching the reference.'

RESPONSE:
{
  "thought": "A is inverted, B checks only adjacent pairs. C and D are equivalent full pairwise scans; the reference and canonical form is the enumerate double loop, option D.",
  "response": "D"
}

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

- Plan: 'Retrieve the canonical inclusive-clamp reference (clamp to [2,8], keep even numbers over an inclusive range).'
- Evidence: 'lo, hi = max(2, min(a, b)), min(8, max(a, b)); [i for i in range(lo, hi + 1) if i % 2 == 0]  # inclusive of hi; keeps evens'
- Plan: 'Decide which option clamps to [2,8], iterates inclusively, and keeps even values.'
- Evidence: 'A keeps odd values (i % 2 != 0). C uses range(lower, upper) and drops the inclusive 8. D fails to clamp with min(2,...)/max(8,...). B clamps to [2,8] inclusively and keeps evens.'

RESPONSE:
{
  "thought": "A returns odds, C is upper-exclusive and drops 8, D fails to clamp to [2,8]. B clamps inclusively and keeps evens, matching the reference.",
  "response": "B"
}

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

- Plan: 'Retrieve the canonical primality-by-trial-division reference.'
- Evidence: 'def is_prime(n):\\n    if n < 2:\\n        return False\\n    for i in range(2, n):\\n        if n % i == 0:\\n            return False\\n    return True'
- Plan: 'Decide which option is a genuine primality test rather than an odd/mod shortcut.'
- Evidence: 'A (l % 2 != 0) and D (l % 2 == 1) only test oddness; length 9 is odd but not prime. C (l % 3 != 0) is unrelated to primality. B is real trial division over range(2, l).'

RESPONSE:
{
  "thought": "A and D only test oddness, C tests divisibility by 3. B is the genuine trial-division primality test matching the reference.",
  "response": "B"
}
"""


# ---- solver_cot: Plan + Evidence -> {thought, response:<letter>} (fill_in_the_middle)
SOLVER_FEW_SHOT_COT_MIDDLE = """### FILL_IN_THE_MIDDLE FEW-SHOT EXAMPLES (SOLVER) ###

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

- Plan: 'Confirm the canonical form for "x is at least twice y".'
- Evidence: '"x is at least twice y" -> x >= 2 * y  (canonical; keeps the equality case)'
- Plan: 'Decide which option reproduces the canonical `>=` shown after the blank.'
- Evidence: 'A uses `>` (drops equality). C uses `==` (only the exact-double case). D rewrites as `max // 2 >= secondMax` (non-canonical integer division). B is the direct canonical `>=` and matches the shown line.'

RESPONSE:
{
  "thought": "'At least twice' maps to `max >= 2 * secondMax`, the canonical form also shown after the blank. A, C, D change or reformulate the comparison; B matches exactly.",
  "response": "B"
}

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

- Plan: 'Confirm the canonical recursive wrap for makeLargestSpecial.'
- Evidence: 'A balanced block is rebuilt as `\\'1\\' + makeLargestSpecial(inner) + \\'0\\'`; the fill must also close the append( call.'
- Plan: 'Decide which option reproduces the wrap and closes the append() parenthesis, matching the shown continuation.'
- Evidence: 'The shown continuation is `\\'1\\' + self.makeLargestSpecial(s[i + 1:j]) + \\'0\\')`. A drops the leading `\\'1\\'` and mangles to `\\'10\\'`; B prepends `\\'10\\'`; C appends `\\'01\\'`. D reproduces the `\\'1\\'...\\'0\\'` wrap and closes the paren.'

RESPONSE:
{
  "thought": "The fill must reproduce the `'1' + rec + '0'` wrap and close the append() paren, matching the shown continuation. A, B and C mangle the wrap; D matches exactly.",
  "response": "D"
}

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

- Plan: 'Confirm the canonical symmetric reachingPoints return.'
- Evidence: 'The shown second line is `sy == ty and sx <= tx and (tx - sx) % ty == 0`; the fill is its mirror `sx == tx and sy <= ty and (ty - sy) % tx == 0`.'
- Plan: 'Decide which option is the exact symmetric mirror of the shown second line.'
- Evidence: 'A changes `<=`->`<` and `(ty - sy)`->`(ty + sy)`. B uses `(sx - sy)`. D changes `<=`->`<`. C matches the canonical mirror exactly.'

RESPONSE:
{
  "thought": "The fill mirrors the shown second line as `sx == tx and sy <= ty and (ty - sy) % tx == 0`. A and D drop the `<=` boundary, A and B use wrong operands; C matches exactly.",
  "response": "C"
}
"""


# ---- solver_contrastive_cot: CORRECT vs WRONG final response (code_completion) ------
SOLVER_FEW_SHOT_CONTRASTIVE_COT_CODE = """### CODE_COMPLETION FEW-SHOT EXAMPLES (SOLVER, CONTRASTIVE) ###

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

- Plan: 'Retrieve the canonical pairwise reference and decide the matching option.'
- Evidence: 'Reference: full pairwise enumerate scan with `idx != idx2 and abs(elem - elem2) < threshold`. A is inverted, B is adjacent-only; C and D are equivalent, D is the canonical enumerate form.'

[CORRECT TRAJECTORY]
{
  "thought": "A returns False on a far pair (inverted); B via zip compares only adjacent pairs. C and D are equivalent full pairwise scans, and the reference/canonical form is the enumerate double loop D.",
  "response": "D"
}

[WRONG TRAJECTORY]
{
  "thought": "Option B is the shortest, most idiomatic one-liner, and concise code is usually intended, so I will answer B.",
  "response": "B"
}
# PITFALL: chose B for BREVITY. B's `zip(numbers, numbers[1:])` compares only adjacent pairs and
# misses non-neighbour close pairs — a real bug. Concision is not a tie-break when behaviour differs.
# Correct answer: D.

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

- Plan: 'Retrieve the inclusive-clamp reference and decide the matching option.'
- Evidence: 'Reference clamps to [2,8] and iterates `range(lo, hi + 1)` keeping evens. A keeps odds, C is upper-exclusive (drops 8), D fails to clamp, B matches.'

[CORRECT TRAJECTORY]
{
  "thought": "A keeps odds, C drops the inclusive 8 with range(lower, upper), D fails to clamp to [2,8]. B clamps inclusively and keeps evens, matching the reference.",
  "response": "B"
}

[WRONG TRAJECTORY]
{
  "thought": "Both B and C keep evens over the clamped bounds; `range(lower, upper)` reads naturally as 'up to upper' and is cleaner without the +1, so C.",
  "response": "C"
}
# PITFALL: `range(lower, upper)` is upper-EXCLUSIVE, so for (2, 8) it yields [2, 4, 6] and drops the
# required 8. The inclusive `range(lower, upper + 1)` (option B) is needed. Correct answer: B.

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

- Plan: 'Retrieve the trial-division primality reference and decide the matching option.'
- Evidence: 'Reference is `for i in range(2, n): if n % i == 0: return False; return True`. A and D only test oddness, C tests divisibility by 3, B is the real primality test.'

[CORRECT TRAJECTORY]
{
  "thought": "A (l % 2 != 0) and D (l % 2 == 1) only test oddness; length 9 is odd but not prime. C (l % 3 != 0) is unrelated to primality. B is a genuine trial-division primality test.",
  "response": "B"
}

[WRONG TRAJECTORY]
{
  "thought": "The docstring True cases have lengths 5 and 7 (odd) and the False case length 6 (even), so prime length seems to coincide with odd length; A reproduces every example, so A.",
  "response": "A"
}
# PITFALL: fitting only the DOCSTRING examples. Odd != prime: length 9 is odd but not prime, and A
# would wrongly return True. Reason on unseen edge cases. Correct answer: B.
"""


# ---- solver_contrastive_cot: CORRECT vs WRONG final response (fill_in_the_middle) ---
SOLVER_FEW_SHOT_CONTRASTIVE_COT_MIDDLE = """### FILL_IN_THE_MIDDLE FEW-SHOT EXAMPLES (SOLVER, CONTRASTIVE) ###

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

- Plan: 'Confirm the canonical "at least twice" comparison and decide the matching option.'
- Evidence: '"x is at least twice y" -> x >= 2 * y (keeps equality). The line shown after the blank is `max >= 2 * secondMax`. A uses `>`, C uses `==`, D uses integer division; B is the canonical `>=`.'

[CORRECT TRAJECTORY]
{
  "thought": "'At least twice' is `max >= 2 * secondMax`, the canonical form also shown after the blank. A, C, D change or reformulate the comparison; B matches exactly.",
  "response": "B"
}

[WRONG TRAJECTORY]
{
  "thought": "'Twice as much' sounds like a strict multiple, so the max must be more than twice the runner-up; `>` captures 'more than twice', so A.",
  "response": "A"
}
# PITFALL: misread "at LEAST twice" as "strictly more than twice". `>` (A) returns -1 when max is
# exactly 2 * secondMax, which the spec accepts, and contradicts the shown `>=` line. Correct answer: B.

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

- Plan: 'Confirm the recursive wrap and decide the option matching the shown continuation.'
- Evidence: 'A block is rebuilt as `\\'1\\' + rec + \\'0\\'`; the shown continuation `\\'1\\' + self.makeLargestSpecial(s[i + 1:j]) + \\'0\\')` also closes the append( call. A, B, C mangle the wrap; D matches and closes the paren.'

[CORRECT TRAJECTORY]
{
  "thought": "The fill must reproduce the `'1' + rec + '0'` wrap and close the append() paren, matching the shown continuation. A, B and C mangle the wrap; D matches exactly.",
  "response": "D"
}

[WRONG TRAJECTORY]
{
  "thought": "The blank is on its own line, so it is a standalone recursive term; A is the shortest recursive-looking expression, so A.",
  "response": "A"
}
# PITFALL: ignored that the fill must match the shown `'1' + ... + '0')` continuation and close the
# append() parenthesis. A drops the leading `'1'` and mis-wraps as `'10'`. Correct answer: D.

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

- Plan: 'Confirm the symmetric mirror and decide the matching option.'
- Evidence: 'The shown second line `sy == ty and sx <= tx and (tx - sx) % ty == 0` mirrors to `sx == tx and sy <= ty and (ty - sy) % tx == 0`. A changes `<=`->`<` and `-`->`+`; B uses `(sx - sy)`; D changes `<=`->`<`; C matches.'

[CORRECT TRAJECTORY]
{
  "thought": "The fill mirrors the shown second line as `sx == tx and sy <= ty and (ty - sy) % tx == 0`. A and D drop the `<=` boundary, A and B use wrong operands; C matches exactly.",
  "response": "C"
}

[WRONG TRAJECTORY]
{
  "thought": "C and D both use `(ty - sy) % tx == 0`; the strict `<` looks safer against edge cases, so I pick D.",
  "response": "D"
}
# PITFALL: swapped `<=` for `<`, dropping the boundary case sy == ty that the symmetric second line
# (`sx <= tx`) clearly keeps. The canonical mirror uses `<=` -> option C. Correct answer: C.
"""


# =====================================================================================
# FEW-SHOT REGISTRY
# =====================================================================================
# Keys mirror the DS-1000 ReWOO-SGR registry so `few_shot_type` validation in the pipeline
# (PlannerREWOOSGR / SolverREWOOSGR) keeps working. Because one pipeline instance runs BOTH
# sub-splits (code, then middle) with a single few-shot string, each base key carries the
# code AND middle examples. The `*_code` / `*_middle` keys expose each sub-type on its own,
# in case the pipeline is later changed to pick few-shots per sub-type.
FEW_SHOT_REGISTRY = {
    "zero_shot": "",
    # Planner few-shots (prepended to PLANNER_PROMPT).
    "cot": FEW_SHOT_COT_CODE + "\n" + FEW_SHOT_COT_MIDDLE,
    "contrastive_cot": FEW_SHOT_CONTRASTIVE_COT_CODE + "\n" + FEW_SHOT_CONTRASTIVE_COT_MIDDLE,
    "auto_cot": FEW_SHOT_AUTO_COT_CODE + "\n" + FEW_SHOT_AUTO_COT_MIDDLE,
    # Solver few-shots (prepended to SOLVER_PROMPT).
    "solver_cot": SOLVER_FEW_SHOT_COT_CODE + "\n" + SOLVER_FEW_SHOT_COT_MIDDLE,
    "solver_contrastive_cot": SOLVER_FEW_SHOT_CONTRASTIVE_COT_CODE + "\n" + SOLVER_FEW_SHOT_CONTRASTIVE_COT_MIDDLE,
    # Per-sub-type variants.
    "cot_code": FEW_SHOT_COT_CODE,
    "cot_middle": FEW_SHOT_COT_MIDDLE,
    "contrastive_cot_code": FEW_SHOT_CONTRASTIVE_COT_CODE,
    "contrastive_cot_middle": FEW_SHOT_CONTRASTIVE_COT_MIDDLE,
    # "auto_cot_code": FEW_SHOT_AUTO_COT_CODE,
    # "auto_cot_middle": FEW_SHOT_AUTO_COT_MIDDLE,
    "solver_cot_code": SOLVER_FEW_SHOT_COT_CODE,
    "solver_cot_middle": SOLVER_FEW_SHOT_COT_MIDDLE,
    "solver_contrastive_cot_code": SOLVER_FEW_SHOT_CONTRASTIVE_COT_CODE,
    "solver_contrastive_cot_middle": SOLVER_FEW_SHOT_CONTRASTIVE_COT_MIDDLE,
    
}
