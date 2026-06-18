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


# Few-shot examples are not implemented yet; keys mirror the DS-1000 registry so that
# `few_shot_type` validation in the pipeline keeps working. Values are empty for now.
FEW_SHOT_REGISTRY = {
    "zero_shot": "",
    "cot": "",
    "contrastive_cot": "",
    "auto_cot": "",
}
