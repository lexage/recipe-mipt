import argparse
import json
import logging
import os
import threading
import time

from pathlib import Path

from src.benchmarks import DS1000, DataItemDS1000
from src.pipelines.pipeline_builder import PipelineBuilder
from src.pipelines.configs import ConfigLoader
from src.utils.loggers import create_logging
from src.utils.token_tracker import (
    TokenTracker,
    install_patch as install_token_patch,
    set_active as set_active_tracker,
)


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
    return parser.parse_args()


def main():

    args = parse_args()
    num_workers = int(args.num_workers)

    config_dir = Path(args.config)
    config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    bench = DS1000(dataset_path=args.dataset)

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

        def run_pipeline(task: DataItemDS1000):
            task_start = time.time()
            # Re-arm the active tracker for this worker thread (thread-local).
            set_active_tracker(eval_tracker)
            result = pipeline.run(task.prompt)
            task_time = time.time() - task_start
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
        n_tasks = 1000  # DS1000
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
        }
        experiment_dir = _find_new_subdir(save_dir, before_subdirs)
        _write_runtime_stats(experiment_dir, stats)

        logging.info(
            f"CONFIG\t{config_path.name}\t{config_time:.3f}s\t"
            f"init={init_time_s:.2f}s filter={filter_apply_time_s} "
            f"peak_rss={peak_rss_mb}MB"
        )


if __name__ == "__main__":
    main()
