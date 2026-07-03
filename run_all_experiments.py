"""Run the 6 CodeMMLU RAG ablation experiments and collect their results.

Each experiment is a separate `python codemmlutest.py --config <cfg> --name <name>`
process (fully isolated: its own vector DB path, its own results notebook), so
they can run in parallel. The shared vLLM servers (solver :7215, embedder :7216)
are the real bottleneck, so `--jobs` throttles how many run at once.

Outputs (in results/):
  <name>.txt        titled notebook: experiment name + parameters + accuracy tables
  <name>_code.csv   per-task outputs for ds_code
  <name>_middle.csv per-task outputs for ds_middle
  summary.csv       one row per experiment for side-by-side comparison

Usage:
  python run_all_experiments.py                 # all 6, 2 at a time (default)
  python run_all_experiments.py --jobs 1        # sequential (reuses server, lightest)
  python run_all_experiments.py --jobs 6        # all at once (hammers the servers)
  python run_all_experiments.py --only 1 3 5    # a subset by number
"""

import argparse
import os
import subprocess
import sys
import threading
import time

EXPERIMENTS = [
    ("baseline",                       "pipeline_configs/ablation_1_baseline.yaml"),
    ("docrag_api",                     "pipeline_configs/ablation_2_docrag_api.yaml"),
    ("instruct",                       "pipeline_configs/ablation_3_instruct.yaml"),
    ("instruct_full_docs",             "pipeline_configs/ablation_4_instruct_full_docs.yaml"),
    ("instruct_fewshot",               "pipeline_configs/ablation_5_instruct_fewshot.yaml"),
    ("instruct_fewshot_full_docs",     "pipeline_configs/ablation_6_instruct_fewshot_full_docs.yaml"),
    # API-selection retriever families (from the spreadsheet)
    ("api_docrag_simple",              "pipeline_configs/ablation_7_api_docrag_simple.yaml"),
    ("api_corag",                      "pipeline_configs/ablation_8_api_corag.yaml"),
    ("api_instruct",                   "pipeline_configs/ablation_9_api_instruct.yaml"),
    ("api_instruct_fewshot",           "pipeline_configs/ablation_10_api_instruct_fewshot.yaml"),
    # documentation-format ablation (SimpleRetriever fixed; only the doc format varies)
    ("doc_0_baseline",                 "pipeline_configs/ablation_doc_0_baseline.yaml"),
    ("doc_1_apiref",                   "pipeline_configs/ablation_doc_1_apiref.yaml"),
    ("doc_2_examples",                 "pipeline_configs/ablation_doc_2_examples.yaml"),
    ("doc_3_rewrite",                  "pipeline_configs/ablation_doc_3_rewrite.yaml"),
    ("doc_4_fulldoc",                  "pipeline_configs/ablation_doc_4_fulldoc.yaml"),
    ("doc_5_allon",                    "pipeline_configs/ablation_doc_5_allon.yaml"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=2, help="max concurrent experiments (default 2)")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="1-based experiment numbers to run (default: all)")
    ap.add_argument("--stream", action="store_true",
                    help="also echo each experiment's stdout to the console "
                         "(prefixed with its name), in addition to the log file")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    selected = EXPERIMENTS if not args.only else [EXPERIMENTS[i - 1] for i in args.only]

    print(f"Running {len(selected)} experiment(s), up to {args.jobs} at a time.\n")

    running = []   # list of (name, Popen, log_file_handle, thread_or_None)
    queue = list(selected)

    def launch(name, cfg):
        log_path = os.path.join(args.out, f"{name}.log")
        log = open(log_path, "w", encoding="utf-8")
        cmd = [sys.executable, "codemmlutest.py", "--config", cfg, "--name", name, "--out", args.out]
        if args.stream:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )

            def pump(p=proc, lg=log, nm=name):
                for line in p.stdout:            # tee: log file + console (prefixed)
                    lg.write(line)
                    lg.flush()
                    sys.stdout.write(f"[{nm}] {line}")
                    sys.stdout.flush()

            th = threading.Thread(target=pump, daemon=True)
            th.start()
        else:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            th = None
        print(f"  [start] {name}  (pid {proc.pid})  -> {log_path}")
        return (name, proc, log, th)

    t0 = time.time()
    while queue or running:
        while queue and len(running) < args.jobs:
            name, cfg = queue.pop(0)
            running.append(launch(name, cfg))
        still = []
        for name, proc, log, th in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, proc, log, th))
            else:
                if th is not None:
                    th.join(timeout=5)
                log.close()
                status = "OK" if rc == 0 else f"FAILED (exit {rc})"
                print(f"  [done ] {name}: {status}  ({time.time() - t0:.0f}s elapsed)")
        running = still
        if queue or running:
            time.sleep(2)

    print(f"\nAll experiments finished in {time.time() - t0:.0f}s.\n")

    # combined summary
    summary = os.path.join(args.out, "summary.csv")
    if os.path.exists(summary):
        print("=== summary.csv ===")
        print(open(summary, encoding="utf-8").read())
    else:
        print("No summary.csv produced — check the per-experiment .log files for errors.")


if __name__ == "__main__":
    main()
