"""Run agentic pipelines (ReAct / ReWOO / MARS / ...) on the CodeMMLU benchmark.

Combines:
  * CodeMMLU dataset loading + multiple-choice evaluation (from `codemmlutest.py`);
  * the config-folder iteration structure of `run_ds1000.py` — every *.yaml in the
    given directory is processed sequentially, each with its own log file and
    result folder.

Unlike `codemmlutest.py` (the SIMPLE baseline) this runner does NOT prepend the
CODE_PROMPT / MIDDLE_PROMPT instructions: the agentic pipelines already carry their
own multiple-choice system prompt (e.g. REACT_SYSTEM_PROMPT from the `codemmlu`
prompt package). The task we hand to the pipeline is just the problem + the four
options + a short per-kind steering hint.

The ReAct agent keeps mutable per-run state on `self`, so a single pipeline
instance is NOT safe to share across threads. To run with several workers we
therefore build one independent pipeline per worker (a small pool) and hand each
task a free pipeline — never sharing one instance between concurrent tasks.

Example:
    python codemmlu_agent_pipelines.py -c my_pipeline_configs/codemmlu_agent_pipelines -n 4
"""

import argparse
import logging
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.pipelines.configs import ConfigLoader
from src.pipelines.pipeline_builder import PipelineBuilder
from src.utils.loggers import create_logging


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


# Short per-kind steering hint appended to the task. The agent's own system prompt
# already describes the full STEP1->STEP2 decision procedure; the hint just shifts
# emphasis to the failure mode that dominates each sub-dataset.
KIND_HINTS = {
    "code": (
        "HINT: the distractors here are usually REAL bugs — prioritise correctness "
        "on the docstring examples and edge cases; use canonical-form tie-breaking "
        "only for genuine ties."
    ),
    "middle": (
        "HINT: the options here are usually functionally EQUIVALENT — behaviour "
        "rarely decides; pick the canonical / minimal reference form."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run agentic pipelines on CodeMMLU")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to a directory with *.yaml configs"
    )
    parser.add_argument(
        "-s", "--save_path", default="results_codemmlu", help="Where to save result CSVs"
    )
    parser.add_argument(
        "-l", "--limit", type=int, default=500,
        help="Max examples per sub-dataset (matches the original 500-row cap)",
    )
    parser.add_argument(
        "-n", "--num_workers", type=int, default=4,
        help="Number of parallel workers (one independent pipeline is built per worker)",
    )
    return parser.parse_args()


def build_agent_task(problem: str, choices, kind: str) -> str:
    """Assemble the multiple-choice task string handed to an agentic pipeline."""
    hint = KIND_HINTS.get(kind, "")
    return (
        f"{hint}\n\n"
        f"Problem: {problem}\n\n"
        f"Solution A: {choices[0]}\n"
        f"Solution B: {choices[1]}\n"
        f"Solution C: {choices[2]}\n"
        f"Solution D: {choices[3]}\n\n"
        "Return ONLY one character (A, B, C, or D) — no explanation, no other words."
    ).strip()


def load_codemmlu(limit: int):
    """Load and prepare the two CodeMMLU sub-datasets (mirrors codemmlutest.py)."""
    ds_middle = load_dataset("Fsoft-AIC/CodeMMLU", "fill_in_the_middle")["test"].to_pandas()
    ds_code = load_dataset("Fsoft-AIC/CodeMMLU", "code_completion")["test"].to_pandas()

    # build_agent_task needs exactly four options.
    ds_middle = ds_middle[ds_middle["choices"].apply(len) == 4]
    ds_code = ds_code[ds_code["choices"].apply(len) == 4]

    if limit:
        ds_middle = ds_middle[:limit]
        ds_code = ds_code[:limit]

    # Middle uses both the question fragment and the problem description.
    ds_middle["input"] = ds_middle.apply(
        lambda row: f"{row['question']} {row['problem_description']}", axis=1
    )
    ds_code["input"] = ds_code["question"]

    return ds_code, ds_middle


def run_split(df: pd.DataFrame, pipeline_pool: "queue.Queue", kind: str,
              num_workers: int) -> pd.DataFrame:
    """Run every row of a sub-dataset through the pipeline pool and score it.

    `pipeline_pool` holds `num_workers` independent pipeline instances. Each task
    checks out a pipeline, runs, and returns it — so no instance is ever used by
    two concurrent tasks at once.
    """
    total = len(df)

    def solve(row):
        task = build_agent_task(row["input"], row["choices"], kind=kind)
        pipeline = pipeline_pool.get()
        task_start = time.time()
        try:
            answer = pipeline.run(task)
        finally:
            pipeline_pool.put(pipeline)
        answer = answer.strip() if isinstance(answer, str) else answer
        logging.info(
            f"TASK\t{kind}\t{row.get('task_id', 'N/A')}\t{time.time() - task_start:.3f}s\t-> {answer}"
        )
        return answer

    # Preserve row order: collect outputs into a slot keyed by position.
    rows = [row for _, row in df.iterrows()]
    outputs = [None] * total
    done = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(solve, row): i for i, row in enumerate(rows)}
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
            done += 1
            if done % 25 == 0 or done == total:
                print(f"\t\t{kind}: {done}/{total}")

    df = df.copy()
    df["output"] = outputs
    df["accuracy"] = (df["output"] == df["answer"]).astype(int)
    return df


def main():
    args = parse_args()

    config_dir = Path(args.config)
    config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    if not config_files:
        raise ValueError(f"No *.yaml configs found in {config_dir}")

    print("Loading CodeMMLU dataset ...")
    ds_code, ds_middle = load_codemmlu(args.limit)
    print(f"\t- code_completion: {len(ds_code)} rows, fill_in_the_middle: {len(ds_middle)} rows")

    for idx, config_path in enumerate(config_files):

        print(f"- processing file {idx + 1}/{len(config_files)}")
        print(f"\t - config: {config_path.name}")

        config_start = time.time()

        pipeline_config = ConfigLoader().load_from_yaml(path_to_cfg=config_path)

        if pipeline_config.logs_path:
            create_logging(
                log_path=pipeline_config.logs_path,
                tag=config_path.stem,
                n_workers=args.num_workers,
                route=True,
            )

        # Build one independent pipeline per worker (sequentially — building is not
        # known to be thread-safe) and stash them in a pool the tasks check out from.
        pipeline_pool = queue.Queue()
        for _ in range(args.num_workers):
            pipeline_pool.put(PipelineBuilder().build(pipeline_config))

        res_code = run_split(ds_code, pipeline_pool, kind="code", num_workers=args.num_workers)
        res_middle = run_split(ds_middle, pipeline_pool, kind="middle", num_workers=args.num_workers)

        save_dir = os.path.join(args.save_path, config_path.stem)
        os.makedirs(save_dir, exist_ok=True)
        res_code.to_csv(os.path.join(save_dir, "ds_code.csv"), encoding="utf-8", index=False)
        res_middle.to_csv(os.path.join(save_dir, "ds_middle.csv"), encoding="utf-8", index=False)

        acc_code = res_code["accuracy"].mean()
        acc_middle = res_middle["accuracy"].mean()
        config_time = time.time() - config_start

        summary = (
            f"CONFIG\t{config_path.name}\t{config_time:.3f}s\t"
            f"code_acc={acc_code:.4f}\tmiddle_acc={acc_middle:.4f}"
        )
        print(f"\t - {summary}")
        logging.info(summary)


if __name__ == "__main__":
    main()
