"""Assemble results/doc_format_ablation.csv from results/summary.csv.

Each experiment run appends a row to results/summary.csv (experiment name +
code_overall + middle_overall). This script pulls the LATEST row per doc-format
experiment and writes the ablation sub-table in the reporting format, with the
accuracy columns filled in. Rows with no run yet stay blank. Idempotent — re-run
after each experiment to refresh.
"""
import csv
import os

SUMMARY = "C:/Projects/recipe-mipt/results/summary.csv"
OUT = "C:/Projects/recipe-mipt/results/doc_format_ablation.csv"

# experiment name -> (DB scope, chunk content, doc rewrite, granularity,
#                     return_examples, merge_examples, return_full_docs)
ROWS = [
    ("doc_0_baseline", "full docs",         "theory only",       "raw",         "chunk",          "FALSE", "FALSE", "FALSE"),
    ("doc_1_apiref",   "API reference only", "theory only",       "raw",         "chunk",          "FALSE", "FALSE", "FALSE"),
    ("doc_2_examples", "full docs",         "theory + examples", "raw",         "chunk",          "TRUE",  "TRUE",  "FALSE"),
    ("doc_3_rewrite",  "full docs",         "theory only",       "LLM-rewrite", "chunk",          "FALSE", "FALSE", "FALSE"),
    ("doc_4_fulldoc",  "full docs",         "theory only",       "raw",         "whole document", "FALSE", "FALSE", "TRUE"),
    ("doc_5_allon",    "API reference only", "theory + examples", "LLM-rewrite", "whole document", "TRUE",  "TRUE",  "TRUE"),
]

# latest code/middle accuracy per experiment name
latest = {}
if os.path.exists(SUMMARY):
    for row in csv.DictReader(open(SUMMARY, encoding="utf-8")):
        latest[row["experiment"]] = (row.get("code_overall", ""), row.get("middle_overall", ""))

header = ["Benchmark", "Model", "api", "Retriever", "Assembler", "DB scope",
          "Chunk content", "Doc rewrite", "Granularity",
          "return_examples", "merge_examples", "return_full_docs", "top_k",
          "CODE acc", "MIDDLE acc", "delta CODE vs baseline"]

base_code = latest.get("doc_0_baseline", ("", ""))[0]
try:
    base_code_f = float(base_code)
except (TypeError, ValueError):
    base_code_f = None

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for name, scope, content, rewrite, gran, ex, merge, full in ROWS:
        code, mid = latest.get(name, ("", ""))
        delta = ""
        if base_code_f is not None and code not in ("", None):
            try:
                delta = f"{float(code) - base_code_f:+.4f}"
            except ValueError:
                delta = ""
        w.writerow(["CodeMMLU", "Qwen/Qwen2.5-32B-Instruct", "completions",
                    "SimpleRetriever", "SimpleContextAssembler", scope, content,
                    rewrite, gran, ex, merge, full, 5, code, mid, delta])

filled = sum(1 for name, *_ in ROWS if latest.get(name, ("", ""))[0])
print(f"wrote {OUT}  ({filled}/{len(ROWS)} rows have numbers)")
