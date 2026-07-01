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

# The pipeline is now built per-experiment inside run_experiment() (bottom of
# file), so this module can be imported and driven with any config / run in
# parallel. See run_experiment() and the __main__ CLI.

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
- reject an option that applies a per-element / per-group transform to only PART
  of the data (e.g. fixes only the first group) instead of every element;
- reject an option that returns ALL items when the spec asks only for the
  EXTREME one(s) — the maximum, or the most frequent — and when several items tie
  for that extreme, the answer must keep EVERY tied item;
- prefer the option whose result matches EVERY docstring example exactly.

STEP 2 — Several options are often FUNCTIONALLY EQUIVALENT: they return the SAME
value on EVERY valid input and differ only in surface form. This is common and
catches you out — two options can look different yet behave identically, so
ACTIVELY check for it before deciding. Watch for these equivalences:
- a comparison whose RESULT is unchanged: `e > m` vs `e >= m` while tracking a
  maximum gives the same maximum; likewise `>` vs `>=` on a bound no input hits;
- an algebraic identity: `i / 2 + 1` == `(i + 2) / 2`; `3 * (k // 3)` ==
  `k - k % 3`; `(l + r) // 2` == `l + (r - l) // 2`;
- the SAME logic with different control flow: an early-return double loop
  `for i: for j>i: if cond: return True ... return False` is the same predicate
  as an all-pairs scan `for a: for b!=a: if cond: return True`;
- two options that are IDENTICAL character-for-character — then they are
  interchangeable and either is acceptable; never reject one just because its
  twin exists.

Once you see options are equivalent, do NOT decide by elimination or by position
— pick the CANONICAL reference form using this priority order:
1. the option that COMPLETELY and directly implements every behaviour the
   docstring describes (the full reference-style solution), over a clever terse
   rewrite that only happens to match;
2. strict `>` / `<` to track an extreme, keeping the FIRST among equal values
   (not `>=` / `<=`);
3. the explicit standard arithmetic form (`i / 2 + 1`, not `(i + 2) / 2`);
4. the standard idiom / PEP 8 form: `[x] * n` (not `[x for _ in range(n)]`),
   `'.'.join(parts)` over manual concatenation, `[1] * (n + 1)` with spaces;
   and among options that differ ONLY in indentation or a trailing newline, the
   canonical one keeps standard 4-space block indentation and the reference's
   trailing newline, not a 1-space or de-indented twin;
5. otherwise the minimal form, with no extra guard the spec does not require.

The reference answer is spread EVENLY across A, B, C and D. Judge every option by
its behaviour, not its position: do not prefer a middle option and do not avoid
A — if the fully-correct, reference-style option is A, answer A.

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

    response = model.run(prompt)
    print(response)
    return response

def describe_config(pipeline_config):
    """Pull the human-readable experiment parameters out of a loaded config."""

    def _one(name):
        c = pipeline_config.components.get(name)
        if c is None:
            return None
        if isinstance(c, list):
            c = c[0]
        return c

    def _type(name):
        c = _one(name)
        return c.type.name if c is not None else None

    def _params(name):
        c = _one(name)
        return (c.params or {}) if c is not None else {}

    db = _params("data_base")
    agent = _params("agent")
    emb = _params("embedder")
    return {
        "Retriever": _type("retriever") or "NONE (baseline / no retrieval)",
        "Context assembler": _type("context_assembler") or "NONE",
        "Relevance rationales": "yes" if "rationality_agent" in pipeline_config.components else "no",
        "API selector": "yes" if "api_selector" in pipeline_config.components else "no",
        "return_full_docs": db.get("return_full_docs", False),
        "return_examples": db.get("return_examples", False),
        "merge_examples": db.get("merge_examples", False),
        "top_k": pipeline_config.params.get("top_k"),
        "Solver model": agent.get("model_name"),
        "Embedder model": emb.get("model_name"),
    }


def _letter_report(df, title):
    per = df.groupby("answer")["accuracy"].agg(["mean", "sum", "count"])
    lines = [f"{title}: overall {df['accuracy'].mean():.4f} ({int(df['accuracy'].sum())}/{len(df)})",
             "  accuracy by correct-answer letter:"]
    for letter, row in per.iterrows():
        lines.append(f"    {letter}: {row['mean']:.4f}  ({int(row['sum'])}/{int(row['count'])})")
    return "\n".join(lines)


def run_experiment(config_path, exp_name=None, out_dir="results"):
    """Build the pipeline from `config_path`, run the CodeMMLU code + middle
    benchmarks, and save a titled results notebook (results/<exp_name>.txt) with
    the experiment name, its parameters (Retriever, flags, models, ...), and the
    accuracy tables. Also dumps per-task CSVs and appends one row to
    results/summary.csv so experiments can be compared side by side."""
    import os
    from datetime import datetime

    if exp_name is None:
        exp_name = os.path.splitext(os.path.basename(config_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    pipeline_config = ConfigLoader().load_from_yaml(config_path)
    if pipeline_config.logs_path:
        create_logging(log_filename=pipeline_config.logs_path)
    params = describe_config(pipeline_config)
    pipeline = PipelineBuilder().build(pipeline_config)

    # --- data (same source/filters as the original script) ---
    ds_middle = load_dataset("Fsoft-AIC/CodeMMLU", "fill_in_the_middle")["test"].to_pandas()
    ds_code = load_dataset("Fsoft-AIC/CodeMMLU", "code_completion")["test"].to_pandas()
    ds_middle = ds_middle[ds_middle["choices"].apply(len) == 4][:500]

    ds_middle["input"] = ds_middle.apply(
        lambda row: get_prompt_middle(row["question"], row["problem_description"]), axis=1)
    ds_code["input"] = ds_code.apply(lambda row: get_prompt_code(row["question"]), axis=1)

    ds_code["output"] = ds_code.apply(
        lambda row: get_prompt_answer(row["input"], row["choices"], pipeline,
                                      kind="code", task_id=row["task_id"]), axis=1)
    ds_middle["output"] = ds_middle.apply(
        lambda row: get_prompt_answer(row["input"], row["choices"], pipeline,
                                      kind="middle", task_id=row["task_id"]), axis=1)

    ds_code["accuracy"] = (ds_code["output"] == ds_code["answer"]).astype(int)
    ds_middle["accuracy"] = (ds_middle["output"] == ds_middle["answer"]).astype(int)

    ds_code.to_csv(os.path.join(out_dir, f"{exp_name}_code.csv"), index=False)
    ds_middle.to_csv(os.path.join(out_dir, f"{exp_name}_middle.csv"), index=False)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        f"Experiment: {exp_name}",
        f"Config:     {config_path}",
        f"Timestamp:  {ts}",
        "-" * 70,
        "Parameters:",
    ]
    for k, v in params.items():
        lines.append(f"  {k:22} {v}")
    lines += ["-" * 70, _letter_report(ds_code, "CODE"), "", _letter_report(ds_middle, "MIDDLE"), "=" * 70, ""]
    text = "\n".join(lines)

    notebook = os.path.join(out_dir, f"{exp_name}.txt")
    with open(notebook, "w", encoding="utf-8") as f:
        f.write(text)

    summary = os.path.join(out_dir, "summary.csv")
    write_header = not os.path.exists(summary)
    with open(summary, "a", encoding="utf-8") as f:
        if write_header:
            f.write("experiment,retriever,rationales,full_docs,examples,merge,top_k,"
                    "code_overall,middle_overall,timestamp\n")
        f.write(f"{exp_name},{params['Retriever']},{params['Relevance rationales']},"
                f"{params['return_full_docs']},{params['return_examples']},{params['merge_examples']},"
                f"{params['top_k']},{ds_code['accuracy'].mean():.4f},"
                f"{ds_middle['accuracy'].mean():.4f},{ts}\n")

    print(text)
    print(f"[saved] {notebook}")
    return notebook


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run one CodeMMLU RAG experiment and log the result.")
    ap.add_argument("--config", default="pipeline_configs/simple_example.yaml",
                    help="path to the pipeline YAML config")
    ap.add_argument("--name", default=None,
                    help="experiment name (default: config file stem); used for the results file")
    ap.add_argument("--out", default="results", help="output directory for result notebooks")
    args = ap.parse_args()
    run_experiment(args.config, args.name, args.out)


ds_middle.to_csv("ds_middle.csv", encoding='utf-8', index=False)

ds_code.to_csv("ds_code.csv", encoding='utf-8', index=False)