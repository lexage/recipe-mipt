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

This runner is SINGLE-THREADED: it builds exactly one pipeline instance and runs
every task through it sequentially. The ReAct agent keeps mutable per-run state on
`self`, and the local Qdrant vector store holds an exclusive file lock on its
storage folder, so a single pipeline instance must not be shared across threads or
duplicated across the same storage path. Running sequentially side-steps both
issues entirely.

Each config is fully isolated: its pipeline is built, used, and then explicitly
closed before the next config is processed, so the Qdrant storage lock is always
released. A failure on one config is logged and skipped instead of aborting the
whole run.

Example:
    python codemmlu_agent_pipelines.py -c my_pipeline_configs/codemmlu_agent_pipelines
"""

import argparse
import gc
import io
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from qdrant_client import QdrantClient
from tqdm import tqdm


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


class _TqdmToLogger(io.StringIO):
    """File-like object that mirrors tqdm's output into the logging system.

    tqdm writes its progress bar to a stream (stderr by default). By pointing it at
    this buffer we capture each refreshed bar line and emit it via logging at INFO
    level, so the progress shows up in the per-config log file too. We also forward
    to the real stderr so the live bar still appears in the terminal.
    """

    def __init__(self, logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level
        self.buf = ""

    def write(self, buf):
        # tqdm uses \r to redraw the bar in place; keep only the meaningful text.
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        if self.buf:
            try:
                self.logger.log(self.level, self.buf)
            except (ValueError, OSError):
                # logging уже закрыт (logging.shutdown() на этапе завершения)
                pass
            # Keep the live, in-place bar visible in the terminal too.
            try:
                sys.stderr.write("\r" + self.buf)
                sys.stderr.flush()
            except (BrokenPipeError, ValueError, OSError):
                # stderr закрыт/сломан при завершении процесса — писать некуда
                pass


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
        # "-l", "--limit", type=int, default=2,
        help="Max examples per sub-dataset (matches the original 500-row cap)",
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
        # ds_code is intentionally NOT capped: code_completion is small (~164 rows)
        # and we want to evaluate on all of it.
        # ds_code = ds_code[:limit]

    logging.info(
            f"ds_code: {len(ds_code)}, ds_middle: {len(ds_middle)}"
        )

    # Middle uses both the question fragment and the problem description.
    ds_middle["input"] = ds_middle.apply(
        lambda row: f"{row['question']} {row['problem_description']}", axis=1
    )
    ds_code["input"] = ds_code["question"]
    

    return ds_code, ds_middle


def _iter_qdrant_clients(root, _seen=None, _depth=0, _max_depth=12):
    """Walk the pipeline object graph and yield every live QdrantClient instance.

    The local Qdrant store keeps an exclusive file lock on its storage folder for
    as long as its client object is alive, but that client is buried deep in the
    graph (e.g. pipeline.agent -> tools -> DBSearchTool.db -> LocalDB.vdb_adapter
    -> QdrantDocsAdapter.client). Shallow attribute guessing misses it, so we
    traverse object __dict__s and common containers, guarding against reference
    cycles via an id()-visited set and a depth cap.
    """
    if _seen is None:
        _seen = set()

    if root is None or _depth > _max_depth:
        return

    obj_id = id(root)
    if obj_id in _seen:
        return
    _seen.add(obj_id)

    if isinstance(root, QdrantClient):
        yield root
        return

    # Skip primitives — they hold no references worth walking and recursing into
    # giant strings/arrays would be pointless and slow.
    if isinstance(root, (str, bytes, int, float, bool, type(None))):
        return

    if isinstance(root, dict):
        children = list(root.keys()) + list(root.values())
    elif isinstance(root, (list, tuple, set, frozenset)):
        children = list(root)
    else:
        children = list(getattr(root, "__dict__", {}).values())

    for child in children:
        yield from _iter_qdrant_clients(child, _seen, _depth + 1, _max_depth)


def close_pipeline(pipeline) -> None:
    """Best-effort release of pipeline resources (Qdrant client / storage lock).

    The local Qdrant store keeps an exclusive lock on its storage folder for as
    long as its client object is alive. If we do not release it before building
    the next config's pipeline, that next build fails on the still-held lock.

    In these REACT/critic configs the LocalDB (and its QdrantClient) is built by
    the pipeline builder but is NOT a dependency of the agent or any tool, so it is
    unreachable from the returned `pipeline` object — walking the pipeline graph
    alone finds nothing to close (that was the "closed 0 QdrantClient(s)" bug). We
    therefore also scan every live object via gc.get_objects() for QdrantClient
    instances. That catches the orphaned client and releases the storage lock
    before the next config opens the same path.
    """
    # Collect clients reachable from the pipeline graph (fast path)...
    clients = {}
    if pipeline is not None:
        for client in _iter_qdrant_clients(pipeline):
            clients[id(client)] = client

    # ...and any still-alive QdrantClient anywhere (catches the orphaned db). This
    # runner is single-threaded and sequential, so the only live client at teardown
    # belongs to the config we are tearing down — closing them all is exactly right.
    for obj in gc.get_objects():
        if isinstance(obj, QdrantClient):
            clients[id(obj)] = obj

    closed = 0
    for client in clients.values():
        try:
            client.close()
            closed += 1
        except Exception:
            logging.exception("QdrantClient.close() failed during cleanup")

    logging.info(f"close_pipeline: closed {closed} QdrantClient(s)")


def _score(df: pd.DataFrame, outputs: list) -> pd.DataFrame:
    """Attach `output`/`accuracy` columns to the first len(outputs) rows of df."""
    scored = df.iloc[: len(outputs)].copy()
    scored["output"] = outputs
    scored["accuracy"] = (scored["output"] == scored["answer"]).astype(int)
    return scored


def run_split(df: pd.DataFrame, pipeline, kind: str, out_csv: str = None) -> pd.DataFrame:
    """Run every row of a sub-dataset through the single pipeline and score it.

    Results are flushed to `out_csv` after every task, so a hard kill / Ctrl-C
    mid-split (the run is hours long) leaves a CSV with every completed row instead
    of discarding all of them. Each write rewrites the whole file with the rows done
    so far — cheap (milliseconds) next to ~50s per task.

    The tqdm progress bar is mirrored into the logging system via _TqdmToLogger, so
    the per-config log file records progress alongside the per-task lines. We cap
    the refresh rate (mininterval) so the bar does not spam the log between the
    ~50s-long tasks.
    """
    total = len(df)
    outputs = []


    tqdm_out = _TqdmToLogger(logging.getLogger())
    bar = tqdm(df.iterrows(), total=total, desc=kind,
               file=tqdm_out, mininterval=5.0, leave=False)
    for done, (_, row) in enumerate(bar, start=1):
        task = build_agent_task(row["input"], row["choices"], kind=kind)
        task_start = time.time()
        try:
            answer = pipeline.run(task)
            answer = answer.strip() if isinstance(answer, str) else answer
        except Exception:
            # Isolate per-task failures: one bad/unparseable LLM response must not
            # abort the whole config and discard every other row's result. Log the
            # full traceback, record a sentinel answer, and keep going.
            logging.exception(
                f"TASK FAILED\t{kind}\t{row.get('task_id', 'N/A')}\t"
                f"{time.time() - task_start:.3f}s"
            )
            answer = "ERROR"
        outputs.append(answer)
        logging.info(
            f"TASK\t{kind}\t{row.get('task_id', 'N/A')}\t{time.time() - task_start:.3f}s\t-> {answer}"
        )
        if out_csv:
            _score(df, outputs).to_csv(out_csv, encoding="utf-8", index=False)
        if done % 25 == 0 or done == total:
            print(f"\t\t{kind}: {done}/{total}")

    bar.close()
    return _score(df, outputs)


def process_config(config_path: Path, ds_code, ds_middle, save_path: str) -> None:
    """Build, run, and tear down a single config end-to-end."""
    config_start = time.time()

    pipeline_config = ConfigLoader().load_from_yaml(path_to_cfg=config_path)

    if pipeline_config.logs_path:
        create_logging(
            log_path=pipeline_config.logs_path,
            tag=config_path.stem,
            n_workers=1,
            route=True,
        )

    # Build a single pipeline instance and run everything through it sequentially.
    # One instance => one Qdrant client => no storage-folder lock contention.
    pipeline = PipelineBuilder().build(pipeline_config)

    save_dir = os.path.join(save_path, config_path.stem)
    os.makedirs(save_dir, exist_ok=True)

    try:
        # run_split flushes each CSV after every task, so even a hard kill mid-split
        # leaves the rows completed so far on disk instead of losing the whole run.
        res_code = run_split(
            ds_code, pipeline, kind="code",
            out_csv=os.path.join(save_dir, "ds_code.csv"),
        )
        res_middle = run_split(
            ds_middle, pipeline, kind="middle",
            out_csv=os.path.join(save_dir, "ds_middle.csv"),
        )

        acc_code = res_code["accuracy"].mean()
        acc_middle = res_middle["accuracy"].mean()
        config_time = time.time() - config_start

        summary = (
            f"CONFIG\t{config_path.name}\t{config_time:.3f}s\t"
            f"code_acc={acc_code:.4f}\tmiddle_acc={acc_middle:.4f}"
        )
        print(f"\t - {summary}")
        logging.info(summary)
    finally:
        # Always release the Qdrant storage lock before the next config builds
        # its own pipeline, even if this config errored mid-run.
        close_pipeline(pipeline)
        del pipeline
        gc.collect()


def main():
    args = parse_args()

    config_dir = Path(args.config)
    config_files = sorted(
        list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    )
    if not config_files:
        raise ValueError(f"No *.yaml configs found in {config_dir}")

    print("Loading CodeMMLU dataset ...")
    ds_code, ds_middle = load_codemmlu(args.limit)
    print(f"\t- code_completion: {len(ds_code)} rows, fill_in_the_middle: {len(ds_middle)} rows")

    failures = []
    for idx, config_path in enumerate(config_files):

        print(f"- processing file {idx + 1}/{len(config_files)}")
        print(f"\t - config: {config_path.name}")

        try:
            process_config(config_path, ds_code, ds_middle, args.save_path)
        except Exception:
            # Log the full traceback and keep going with the remaining configs
            # instead of aborting the entire run on a single bad config.
            logging.exception(f"Config {config_path.name} failed")
            print(f"\t - FAILED: {config_path.name} (see logs for traceback)")
            failures.append(config_path.name)
            # A failure may have happened AFTER the QdrantClient was opened but
            # before process_config's own finally ran (e.g. build error past the
            # db step). That orphaned client still holds the storage lock, so the
            # next config opening the same path would fail. close_pipeline(None)
            # scans every live object via gc and closes any open client.
            close_pipeline(None)

    if failures:
        print(f"\nDone with {len(failures)} failed config(s): {', '.join(failures)}")
    else:
        print("\nAll configs processed successfully.")


if __name__ == "__main__":
    main()
    # fastembed/TensorFlow/onnxruntime spawn non-daemon background threads that
    # keep the interpreter alive after main() returns. A lingering process keeps
    # any still-open Qdrant client — and thus the storage-folder lock — held,
    # which makes the NEXT run fail to acquire the lock. Flush stdio and hard-exit
    # so those threads cannot outlive the script.
    logging.shutdown()  # flush + close file handlers (os._exit skips atexit)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)