from datasets import load_dataset
import pandas as pd
from src.pipelines.configs import (
    PipelineConfig,
    ComponentConfig,
)
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel, SecretStr
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
import json
from src.pipelines.configs import ConfigLoader
from src.utils.loggers import create_logging

base_url = "http://localhost:8080/vllm/qwen35-moe/v1"
model= "Qwen/Qwen3-Coder-Next"
llm_client = ChatOpenAI(
            model=model,  # type: ignore[unknown-argument]
            api_key=SecretStr("-"),  # type: ignore[unknown-argument]
            base_url=str(base_url),  # type: ignore[unknown-argument]
            # seed для обеспечения воспроизводимости ответов
            seed=42,
        )
from src.pipelines.constants import ComponentNames, PipelinesNames

from src.pipelines.pipeline_builder import PipelineBuilder
import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

pipeline_config = ConfigLoader().load_from_yaml("/workspace/proj/grant/exps_17_06_26_codemmlu/simple_test_codemmlu.yaml")

if pipeline_config.logs_path:
    create_logging(log_filename=pipeline_config.logs_path)

pipeline = PipelineBuilder().build(pipeline_config)

def check(correct, output):
    if (correct == output):
        return 1
    return 0

def get_prompt_code( question):
    prompt = question
    return prompt

def get_prompt_middle( question, question2):
    prompt = question + " " + question2
    return prompt

# --- Point 2: dataset-specific prompts -------------------------------------
# Code (HumanEval-style) failures are mostly REAL bugs -> short, reasoning /
# edge-case focused; NO style rules (they distract and hurt this set).
CODE_PROMPT = """You are answering a multiple-choice question about Python code. You are given a
function (with its docstring and examples) and four candidate completions
A, B, C, D. Exactly one is the reference-correct answer.

Reason SILENTLY; output exactly ONE character — A, B, C, or D. No words, no code.

STEP 1 — Prefer the option that is ACTUALLY CORRECT on every input the docstring
implies:
- mentally run each option on the docstring examples AND on the boundary cases:
  empty input, a single element, 0, 1, negative numbers, duplicates, first/last
  index;
- reject a wrong comparison / operator / variable or an off-by-one (e.g. a
  primality bound that must reach int(n**0.5)+1; ">" vs ">=" or "<0" vs "<=0"
  where it changes the result);
- reject an option that DROPS a guard the spec needs (empty list -> return the
  sentinel such as nan / [] / None BEFORE dividing or indexing), or that
  DEDUPLICATES data that must keep duplicates;
- reject a wrong formula, wrong tie-breaking, or a different algorithm;
- prefer the option whose result matches EVERY docstring example exactly.

STEP 2 — If two or more options are FUNCTIONALLY EQUIVALENT (identical result on
every input), pick the CANONICAL / minimal form — the one identical, character
for character, to the simplest reference form:
- strict `>` to track a maximum (not `>=`); keep the FIRST among equal values;
- minimal form: no extra guard the spec does not need (e.g. no `if max == min`
  branch when the input is guaranteed to have >= 2 distinct values);
- standard 4-space indentation; PEP 8 spacing `[1] * (n + 1)` (not `[1]*(n+1)`);
- `[x] * n` over `[x for _ in range(n)]`; `'.'.join(parts)` over manual concat;
- arithmetic identities are equal — `i / 2 + 1` == `(i + 2) / 2`,
  `3 * (k // 3)` == `k - k % 3` — choose the explicit reference form / shortest;
- prefer the concise idiom the reference uses (e.g. `decode = encode(encode(s))`
  over re-implementing the transform).

Output a single letter only."""

# Middle (fill-in-the-blank) failures are mostly EQUIVALENT choices ->
# concise canonical-form rules + few-shot demonstrations.
MIDDLE_PROMPT = """You are answering a multiple-choice question about Python code. You are given a
code fragment (sometimes with one missing line) and four options A, B, C, D.
Exactly one is the reference-correct answer.

Reason SILENTLY; output exactly ONE character — A, B, C, or D. No words, no code.

STEP 1 — Discard options that change BEHAVIOUR: wrong operator/variable/off-by-one;
parentheses that change precedence so the result changes (Python: `and` binds
tighter than `or`, so `a and b or c` != `a and (b or c)`); a wrong boundary like
`<0` vs `<=0`; an edge case handled differently.

STEP 2 — If two or more options are FUNCTIONALLY EQUIVALENT, choose the CANONICAL
form (the one identical, character for character, to the simplest reference form):
- midpoint `(l + r) // 2`   (not `l+(r-l)//2`, `(l+r)>>1`, `(r-l)//2+l`)
- infinity `math.inf` / `-math.inf`   (not `float('inf')`, `sys.maxsize`)
- repeated list `[x] * n`   (not `[x for _ in range(n)]`, `n*[x]`)
- bare tuple `return a, b`   (not `return [a, b]`)
- minimal form: no extra guards/casts/parens (`1 << 31 - i`, not `1 << (31 - i)`)
- set a bit with OR `x | 1 << i`   (not `x ^ 1 << i`)
- `.remove(x)` when the element is present   (not `.discard(x)`)
- boundary `== n`   (not `>= n`) when the index moves by 1 per step
- `1 + max(...)`   (not `max(...) + 1`); `3 * (row // 3)` (not `(row // 3) * 3`)
- shortest idiom: `'.'.join(parts)`, `0 in column`, `functools.reduce(...)` over a loop
- PEP 8 spacing `[1] * (n + 1)` (not `[1]*(n+1)`)

Judge by the FORM, not by the position. Functionally equivalent options are
common here, and the reference (the simplest, most standard form) is FREQUENTLY
option A — never reject an option, and never avoid A, just because of where it
appears. If the cleanest canonical form is option A, answer A.

Worked examples (decide, then output only the letter):
Example 1 — blank is the midpoint.
A `(l + r) // 2`   B `l + (r - l) // 2`   C `(l + r) >> 1`   D `(r - l) // 2 + l`
All four are equal; canonical midpoint is `(l + r) // 2`. Answer: A
Example 2 — initialise "negative infinity".
A `-float('inf')`   B `-1e309`   C `-math.inf`   D `float('-inf')`
All equal; use the math constant. Answer: C
Example 3 — build a size-n list of zeros.
A `[0] * n`   B `[0 for _ in range(n)]`   C `n * [0]`   D `[0]*(n)`
All equal; list multiplication with PEP 8 spacing. Answer: A

Output a single letter only."""


from src.rag.simple.retriever import set_current_task, reset_retrieval_trace


def get_prompt_answer(problem, answers, model, kind="middle", task_id=None):
    base = CODE_PROMPT if kind == "code" else MIDDLE_PROMPT

    prompt = base + f"""

        Problem: {problem}

        Solution A: {answers[0]}
        Solution B: {answers[1]}
        Solution C: {answers[2]}
        Solution D: {answers[3]}
        Return ONLY one character of the right solution (A, B, C, or D) — no explanation, no other words.
    """

    # Tag the retrieval log with this task so we can audit what the retriever
    # actually pulled in for each failing task (Code A focus).
    set_current_task(task_id=task_id, kind=kind)

    response = model.run(prompt)
    print(response)
    return response

# Login using e.g. `huggingface-cli login` to access this dataset
ds_middle = load_dataset("Fsoft-AIC/CodeMMLU", "fill_in_the_middle")

ds_code = load_dataset("Fsoft-AIC/CodeMMLU", "code_completion")

ds_middle = ds_middle["test"].to_pandas()

ds_code = ds_code["test"].to_pandas()

ds_middle = ds_middle[ds_middle['choices'].apply(len) == 4]

ds_middle = ds_middle[: 500]
##ds_code['output'] = ds_code.apply(lambda row: get_prompt_code(row[""], row[""]), axis=1)



ds_middle["input"] = ds_middle.apply(
    lambda row: get_prompt_middle(row["question"], row["problem_description"]), 
    axis=1
)

ds_code["input"] = ds_code.apply(
    lambda row: get_prompt_code(row["question"]), 
    axis=1
)


# Start a fresh retrieval trace for this run
reset_retrieval_trace()

ds_code["output"] = ds_code.apply(
    lambda row: get_prompt_answer(
        row["input"], row["choices"], pipeline,
        kind="code", task_id=row["task_id"],
    ),
    axis=1,
)

ds_middle["output"] = ds_middle.apply(
    lambda row: get_prompt_answer(
        row["input"], row["choices"], pipeline,
        kind="middle", task_id=row["task_id"],
    ),
    axis=1,
)

ds_code['accuracy'] = (ds_code['output'] == ds_code['answer']).astype(int)

ds_middle['accuracy'] = (ds_middle['output'] == ds_middle['answer']).astype(int)


print("Code:")
print(ds_code.groupby("answer")['accuracy'].mean())

print("middle:")
print(ds_middle.groupby("answer")['accuracy'].mean())


ds_middle.to_csv("ds_middle.csv", encoding='utf-8', index=False)

ds_code.to_csv("ds_code.csv", encoding='utf-8', index=False)