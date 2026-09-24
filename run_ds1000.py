import argparse
import json
import logging
import os
import shutil
import threading
import time

from pathlib import Path

# Embedder + LLM are served locally on the server — never reach out to
# huggingface.co (connection resets -> 5 retries per call = minutes wasted).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.benchmarks import DS1000, DataItemDS1000
from src.pipelines.pipeline_builder import PipelineBuilder
from src.pipelines.configs import ConfigLoader
from src.utils.loggers import create_logging
from src.utils.token_tracker import (
    TokenTracker,
    install_patch as install_token_patch,
    set_active as set_active_tracker,
)
from src.utils.retrieval_log import take as take_retrieval_record
from src.utils.prompt_registry import collect_prompts, describe_prompts
from src.agent_constructor.prompts import set_overrides as set_prompt_overrides


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# RSS sampler — measures peak resident memory per config (mb)
# ---------------------------------------------------------------------------

class RSSSampler:
    """Background thread that records peak RSS in MB until `stop()`.

    Falls back to a no-op if psutil isn't installed.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.peak_mb = 0.0
        self._running = False
        self._thread = None

    def _loop(self) -> None:
        proc = psutil.Process()
        while self._running:
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            if rss_mb > self.peak_mb:
                self.peak_mb = rss_mb
            time.sleep(self.interval)

    def start(self) -> None:
        if not _HAS_PSUTIL:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if not _HAS_PSUTIL:
            return None
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return round(self.peak_mb, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_new_subdir(save_path: str, before: set[str]) -> str | None:
    """Return the timestamped sub-folder that bench.eval just created."""
    if not os.path.isdir(save_path):
        return None
    after = set(os.listdir(save_path))
    new = after - before
    if not new:
        return None
    # bench.eval names new dirs as YYYYMMDD_HHMMSS — newest wins on sort.
    newest = sorted(new)[-1]
    return os.path.join(save_path, newest)


def _write_runtime_stats(target_dir: str | None, stats: dict) -> None:
    if not target_dir:
        logging.warning("Cannot write runtime_stats.json: target dir unknown.")
        return
    out_path = os.path.join(target_dir, "runtime_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logging.info("Wrote runtime stats → %s", out_path)


def _snapshot_config(target_dir: str | None, config_path) -> None:
    """Copy the config that produced this run next to its results."""
    if not target_dir:
        return
    try:
        shutil.copyfile(config_path, os.path.join(target_dir, "config.yaml"))
    except OSError as exc:
        logging.warning("Could not snapshot config: %s", exc)


def _write_system_prompt(target_dir: str | None, pipeline) -> None:
    """Save the solver's system prompt exactly as it ran.

    With a rule writer attached the text is assembled at build time and exists
    nowhere else — the config only names the component that wrote it.
    """
    if not target_dir:
        return
    agent = getattr(pipeline, "agent", None)
    text = getattr(agent, "system_prompt", None)
    if not text:
        return
    out_path = os.path.join(target_dir, "system_prompt_used.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    logging.info("Wrote system prompt (%d chars) -> %s", len(text), out_path)


def _write_prompts(target_dir: str | None, pipeline) -> None:
    """Dump the full text of every prompt the pipeline declared.

    Written after eval so query-time prompts have registered (registration is
    lazy — see src/agent_constructor/prompts.py).
    """
    if not target_dir:
        return
    prompts = collect_prompts(pipeline)
    if not prompts:
        return
    payload = {
        key: {
            "name": prompt.name,
            "optimizable": prompt.optimizable,
            "revision": prompt.revision,
            "fingerprint": prompt.fingerprint(),
            "text": prompt.text,
        }
        for key, prompt in prompts.items()
    }
    out_path = os.path.join(target_dir, "prompts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %d prompts → %s", len(payload), out_path)


def _log_retrieval(task, records: list, lock, text_chars: int) -> None:
    """Append this task's retrieval record to `records` (thread-safe).

    Must run on the worker thread that called `pipeline.run` — the pipeline
    stashes the chunks thread-locally (see src/utils/retrieval_log.py).
    """
    record = take_retrieval_record()
    chunks = record["chunks"]
    entry = {
        "problem_id": task.metadata.get("problem_id"),
        "library": task.metadata.get("library"),
        "perturbation_type": task.metadata.get("perturbation_type"),
        "n_chunks": len(chunks),
        "context_chars": len(record["context"] or ""),
        "chunks": [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "score": (chunk.metadata or {}).get("score"),
                "source": (chunk.metadata or {}).get("source"),
                "library": (chunk.metadata or {}).get("library"),
                "chars": len(chunk.text or ""),
                "text": (chunk.text or "") if text_chars <= 0
                        else (chunk.text or "")[:text_chars],
            }
            for chunk in chunks
        ],
    }
    with lock:
        records.append(entry)


def _write_chunk_log(target_dir: str | None, records: list) -> None:
    """Dump the per-task retrieval log (one JSON object per task)."""
    if not records:
        return
    if not target_dir:
        logging.warning("Cannot write retrieved_chunks.jsonl: target dir unknown.")
        return
    out_path = os.path.join(target_dir, "retrieved_chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logging.info("Wrote retrieval log (%d tasks) → %s", len(records), out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline with config")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to config dir",
    )
    parser.add_argument(
        "-d", "--dataset", default="data/ds1000/ds1000.jsonl.gz",
        help="Path to DS1000 dataset",
    )
    parser.add_argument(
        "-s", "--save_path", default="results",
        help="Path to where to save results",
    )
    parser.add_argument(
        "-n", "--num_workers", default=4, help="Number of workers",
    )
    parser.add_argument(
        "-l", "--limit", type=int, default=0,
        help="Run only N tasks, evenly spaced across the dataset (0 = all).",
    )
    parser.add_argument(
        "--log-chunks", action="store_true",
        help="Log the chunks the retriever returned for every task into "
             "retrieved_chunks.jsonl next to runtime_stats.json.",
    )
    parser.add_argument(
        "--log-chunk-chars", type=int, default=800,
        help="Chars of each chunk's text kept in the log (0 = full text).",
    )
    parser.add_argument(
        "--prompts",
        help="JSON of prompt overrides (best_prompts.json from "
             "optimize_prompts.py) applied when each prompt registers.",
    )
    parser.add_argument(
        "--exclude-split",
        help="split.json from optimize_prompts.py — drop its `train` "
             "problem_ids so the score is reported on unseen tasks only.",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    num_workers = int(args.num_workers)

    config_dir = Path(args.config)
    if config_dir.is_dir():
        # sorted() so the run order is reproducible — glob order is arbitrary,
        # and these configs are meant to run in a specific sequence.
        config_files = (sorted(config_dir.glob("*.yaml"))
                        + sorted(config_dir.glob("*.yml")))
    elif config_dir.is_file():
        config_files = [config_dir]
    else:
        raise SystemExit(f"config path not found: {config_dir}")

    bench = DS1000(dataset_path=args.dataset)

    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        # best_prompts.json is {key: text}; prompts.json is {key: {...,"text"}}.
        overrides = {
            key: value["text"] if isinstance(value, dict) else value
            for key, value in overrides.items()
        }
        set_prompt_overrides(overrides)
        print(f"- applying {len(overrides)} prompt override(s) from {args.prompts}")

    if args.exclude_split:
        with open(args.exclude_split, "r", encoding="utf-8") as f:
            excluded = set(json.load(f).get("train", []))
        before = len(bench.dataset._items)
        bench.dataset._items = [
            item for item in bench.dataset._items
            if item.metadata.get("problem_id") not in excluded
        ]
        print(f"- excluding {before - len(bench.dataset._items)} optimizer "
              f"train tasks -> evaluating on {len(bench.dataset._items)}")

    if args.limit and args.limit < len(bench.dataset._items):
        items = bench.dataset._items
        step = len(items) / args.limit
        bench.dataset._items = [items[int(i * step)] for i in range(args.limit)]
        print(f"- limiting eval to {args.limit} tasks (evenly spaced across "
              f"{len(items)})")

    if not _HAS_PSUTIL:
        logging.warning(
            "psutil not installed → peak_rss_mb will be reported as null. "
            "Install with: python -m pip install psutil"
        )

    # Install token-tracking monkey-patch once.
    install_token_patch()

    for idx, config_path in enumerate(config_files):

        print(f"- processing file {idx+1}/{len(config_files)}")
        print(f"\t - config: {config_path.name}")

        config_start = time.time()

        pipeline_config = ConfigLoader().load_from_yaml(path_to_cfg=config_path)

        if pipeline_config.logs_path:
            create_logging(
                log_path=pipeline_config.logs_path,
                tag=config_path.stem,
                n_workers=num_workers,
                route=True,
            )

        # ---------- start memory sampler ----------
        rss = RSSSampler(interval=0.5)
        rss.start()

        # ---------- build pipeline (filter.apply runs here) ----------
        init_tracker = TokenTracker()
        set_active_tracker(init_tracker)
        init_start = time.time()
        try:
            pipeline = PipelineBuilder().build(pipeline_config)
        except Exception as exc:
            set_active_tracker(None)
            rss.stop()
            logging.exception(
                "CONFIG BUILD FAILED — skipping %s", config_path.name
            )
            print(f"\t - SKIPPED (build error): {exc}")
            continue
        init_time_s = time.time() - init_start
        set_active_tracker(None)

        filter_apply_time_s = getattr(pipeline, "_filter_apply_time", None)
        document_filter_apply_time_s = getattr(
            pipeline, "_document_filter_apply_time", None
        )

        # ---------- run benchmark ----------
        save_dir = os.path.join(args.save_path, config_path.stem)
        before_subdirs = (
            set(os.listdir(save_dir)) if os.path.isdir(save_dir) else set()
        )

        chunk_log: list = []
        chunk_log_lock = threading.Lock()

        def run_pipeline(task: DataItemDS1000):
            task_start = time.time()
            # Re-arm the active tracker for this worker thread (thread-local).
            set_active_tracker(eval_tracker)
            result = pipeline.run(task.prompt)
            task_time = time.time() - task_start
            if args.log_chunks:
                # Same thread that ran the pipeline -> gets this task's record.
                _log_retrieval(task, chunk_log, chunk_log_lock, args.log_chunk_chars)
            logging.info(
                f"TASK\t{task.metadata.get('problem_id', 'N/A')}\t{task_time:.3f}s"
            )
            return result

        eval_tracker = TokenTracker()
        set_active_tracker(eval_tracker)
        eval_start = time.time()
        try:
            bench.eval(
                run_method=run_pipeline,
                save_path=save_dir,
                num_workers=num_workers,
            )
        finally:
            pipeline.close()
            set_active_tracker(None)
        total_time_s = time.time() - eval_start

        peak_rss_mb = rss.stop()

        # ---------- assemble and persist stats ----------
        config_time = time.time() - config_start
        n_tasks = len(bench.dataset._items)  # actual tasks run (respects --limit)
        stats = {
            "config_name": config_path.stem,
            "init_time_s": round(init_time_s, 2),
            "filter_apply_time_s": (
                round(filter_apply_time_s, 2)
                if filter_apply_time_s is not None else None
            ),
            "document_filter_apply_time_s": (
                round(document_filter_apply_time_s, 2)
                if document_filter_apply_time_s is not None else None
            ),
            "total_time_s": round(total_time_s, 2),
            "mean_task_time_s": round(total_time_s / n_tasks, 4),
            "peak_rss_mb": peak_rss_mb,
            "config_time_s": round(config_time, 2),

            # Tokens during bench.eval (1000 tasks).
            "eval_input_tokens":       eval_tracker.input_tokens,
            "eval_output_tokens":      eval_tracker.output_tokens,
            "eval_total_tokens":       eval_tracker.total_tokens,
            "eval_mean_input_tokens":  round(eval_tracker.input_tokens  / n_tasks, 2),
            "eval_mean_output_tokens": round(eval_tracker.output_tokens / n_tasks, 2),
            "eval_mean_total_tokens":  round(eval_tracker.total_tokens  / n_tasks, 2),
            "eval_llm_calls":          eval_tracker.n_calls,

            # Tokens during pipeline build / filter.apply (one-time cost).
            "init_input_tokens":       init_tracker.input_tokens,
            "init_output_tokens":      init_tracker.output_tokens,
            "init_total_tokens":       init_tracker.total_tokens,
            "init_llm_calls":          init_tracker.n_calls,

            # Which prompt text produced this run (full text in prompts.json).
            "prompts": describe_prompts(pipeline),
        }
        experiment_dir = _find_new_subdir(save_dir, before_subdirs)
        _write_runtime_stats(experiment_dir, stats)
        _write_chunk_log(experiment_dir, chunk_log)
        _snapshot_config(experiment_dir, config_path)
        _write_prompts(experiment_dir, pipeline)
        _write_system_prompt(experiment_dir, pipeline)

        logging.info(
            f"CONFIG\t{config_path.name}\t{config_time:.3f}s\t"
            f"init={init_time_s:.2f}s filter={filter_apply_time_s} "
            f"peak_rss={peak_rss_mb}MB"
        )


if __name__ == "__main__":
    main()
