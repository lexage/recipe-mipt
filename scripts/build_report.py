#!/usr/bin/env python3
"""Build a single, hand-off analysis document from a DS1000 experiment run.

Merges the three per-run artifacts produced by ``run_ds1000.py``:

    <experiment_dir>/
        results.csv      # score / pass-fail / library / perturbation per problem
        answers.jsonl     # final (post-processed) code per problem
        traces.jsonl      # intermediate artifacts per problem (see src/tracing)

into a single Markdown report (and, optionally, a merged JSON) that can be handed
to an LLM for failure-taxonomy analysis, or read by a human.

The report is *adaptive*: sections that have no data for a given pipeline
configuration (no generated queries, no selected APIs, no rationales, ...) are
omitted automatically, so the same script works for baseline, naive-RAG and
API-instruct runs without any flags.

Examples
--------
    # Only failed problems, truncate long chunk/prompt text (default):
    python scripts/build_report.py results/naive_rag/20260703_120000

    # Everything, one library, cap at 30 problems, also emit merged JSON:
    python scripts/build_report.py results/api_instruct/RUN \\
        --all --library Pandas --limit 30 --json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def _read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run(experiment_dir: str) -> Dict[Any, dict]:
    """Return a mapping ``problem_id -> merged record`` for one experiment dir."""
    results_path = os.path.join(experiment_dir, "results.csv")
    answers_path = os.path.join(experiment_dir, "answers.jsonl")
    traces_path = os.path.join(experiment_dir, "traces.jsonl")

    merged: Dict[Any, dict] = defaultdict(dict)

    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = _coerce_id(row.get("problem_id"))
                merged[pid]["result"] = row
    else:
        print(f"warning: {results_path} not found", file=sys.stderr)

    for row in _read_jsonl(answers_path):
        pid = _coerce_id((row.get("metadata") or {}).get("problem_id"))
        merged[pid]["answer_code"] = row.get("code", "")

    traces = _read_jsonl(traces_path)
    if not traces:
        print(
            f"note: no traces.jsonl in {experiment_dir} "
            "(run was produced before tracing was enabled) — "
            "report will only contain results + final code.",
            file=sys.stderr,
        )
    for row in traces:
        pid = _coerce_id(row.get("problem_id"))
        merged[pid]["trace"] = row

    return dict(merged)


def _coerce_id(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


# --------------------------------------------------------------------------- #
# Filtering / ordering                                                         #
# --------------------------------------------------------------------------- #

def _passed(record: dict) -> Optional[bool]:
    result = record.get("result")
    if not result:
        return None
    score = result.get("score")
    try:
        return float(score) >= 1.0
    except (TypeError, ValueError):
        return None


def select_records(
    merged: Dict[Any, dict],
    only_failed: bool,
    library: Optional[str],
    limit: Optional[int],
) -> List[tuple]:
    items = sorted(merged.items(), key=lambda kv: (str(type(kv[0])), kv[0]))
    selected = []
    for pid, record in items:
        result = record.get("result") or {}
        if library and result.get("library") != library:
            continue
        if only_failed and _passed(record) is True:
            continue
        selected.append((pid, record))
    if limit is not None:
        selected = selected[:limit]
    return selected


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #

def _truncate(text: Optional[str], limit: Optional[int]) -> str:
    if text is None:
        return ""
    text = str(text)
    if limit and len(text) > limit:
        return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"
    return text


def _fence(text: Optional[str], lang: str = "", limit: Optional[int] = None) -> str:
    body = _truncate(text, limit).rstrip("\n")
    return f"```{lang}\n{body}\n```"


def render_overview(merged: Dict[Any, dict], experiment_dir: str) -> str:
    total = len(merged)
    scored = [r for r in merged.values() if _passed(r) is not None]
    passed = sum(1 for r in scored if _passed(r) is True)

    by_lib: Dict[str, List[bool]] = defaultdict(list)
    by_pert: Dict[str, List[bool]] = defaultdict(list)
    for record in merged.values():
        result = record.get("result") or {}
        p = _passed(record)
        if p is None:
            continue
        by_lib[result.get("library", "?")].append(p)
        by_pert[result.get("perturbation_type", "?")].append(p)

    lines = [
        f"# DS1000 run report — `{experiment_dir}`",
        "",
        f"- Problems: **{total}**",
    ]
    if scored:
        lines.append(f"- Overall accuracy: **{passed}/{len(scored)} = {passed / len(scored):.1%}**")

    def _table(title: str, groups: Dict[str, List[bool]]) -> List[str]:
        if not groups:
            return []
        rows = [f"", f"### {title}", "", "| group | n | accuracy |", "|---|---:|---:|"]
        for key in sorted(groups):
            vals = groups[key]
            acc = sum(vals) / len(vals) if vals else 0.0
            rows.append(f"| {key} | {len(vals)} | {acc:.1%} |")
        return rows

    lines += _table("By library", by_lib)
    lines += _table("By perturbation type", by_pert)
    return "\n".join(lines)


def render_problem(pid: Any, record: dict, max_chars: Optional[int]) -> str:
    result = record.get("result") or {}
    trace = record.get("trace") or {}

    passed = _passed(record)
    status = {True: "✅ passed", False: "❌ failed", None: "· unknown"}[passed]
    library = result.get("library", "?")
    pert = result.get("perturbation_type", "?")

    out: List[str] = [f"## Problem {pid} — {library} / {pert} — {status}"]

    if passed is False and result.get("result"):
        out += ["", f"**Error:** `{_truncate(result.get('result'), 400)}`"]

    prompt = trace.get("prompt") or result.get("prompt")
    if prompt:
        out += ["", "**Prompt**", _fence(prompt, "text", max_chars)]

    # Generated search queries — only if they differ from the bare prompt.
    queries = [q for q in (trace.get("queries") or []) if q and q != prompt]
    if queries:
        out += ["", "**Generated queries**"]
        out += [f"- {q}" for q in queries]

    # API selector output (API retrievers only).
    apis = trace.get("apis") or []
    if apis:
        out += ["", "**Selected APIs**"]
        for call in apis:
            selected = ", ".join(f"`{a}`" for a in call.get("selected", [])) or "_none_"
            out.append(f"- selected: {selected}")
            raw = call.get("raw")
            if raw:
                out.append(f"  - raw selector output: {_truncate(' | '.join(map(str, raw)), 300)}")

    # Retrieved chunks (any retriever).
    chunks = trace.get("retrieved_chunks") or []
    if chunks:
        out += ["", f"**Retrieved chunks** ({len(chunks)})"]
        for i, ch in enumerate(chunks):
            meta_bits = []
            for label in ("retrieval_api", "doc_name", "library", "score", "source"):
                if ch.get(label) not in (None, ""):
                    meta_bits.append(f"{label}={ch[label]}")
            header = f"[{i}] " + (" · ".join(meta_bits) if meta_bits else f"id={ch.get('id')}")
            out += [f"- {header}", _fence(ch.get("text", ""), "", max_chars)]

    # Rationales (instruct-rag family) — mirrored for convenience.
    rationales = trace.get("rationales") or []
    if rationales:
        out += ["", f"**Rationales** ({len(rationales)})"]
        for i, r in enumerate(rationales):
            api = r.get("retrieval_api")
            out += [f"- [{i}] {('api=' + str(api)) if api else ''}".rstrip(),
                    _fence(r.get("text", ""), "", max_chars)]

    # Final prompt actually sent to the solver.
    final_prompt = trace.get("final_prompt")
    if final_prompt:
        out += ["", "<details><summary><b>Final prompt sent to model</b></summary>", "",
                _fence(final_prompt, "text", max_chars), "", "</details>"]

    # Raw model answer vs post-processed code.
    raw_answer = trace.get("answer")
    code = record.get("answer_code")
    if raw_answer is not None and raw_answer != code:
        out += ["", "**Raw model answer**", _fence(raw_answer, "python", max_chars)]
    if code is not None:
        out += ["", "**Final code (evaluated)**", _fence(code, "python", max_chars)]

    return "\n".join(out)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment_dir", help="Path to results/<config>/<timestamp>")
    p.add_argument("-o", "--out", default=None, help="Output markdown path (default: <dir>/report.md)")
    p.add_argument("--all", action="store_true", help="Include passed problems too (default: failed only)")
    p.add_argument("--library", default=None, help="Restrict to one library, e.g. Pandas")
    p.add_argument("--limit", type=int, default=None, help="Max number of problems to include")
    p.add_argument("--max-chunk-chars", type=int, default=1200,
                   help="Truncate long text blocks to this many chars (0 = no limit)")
    p.add_argument("--json", action="store_true", help="Also write merged records as <dir>/report.json")
    return p.parse_args()


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.rstrip("/")
    if not os.path.isdir(experiment_dir):
        sys.exit(f"error: not a directory: {experiment_dir}")

    merged = load_run(experiment_dir)
    if not merged:
        sys.exit(f"error: no results/answers/traces found in {experiment_dir}")

    max_chars = args.max_chunk_chars or None
    selected = select_records(merged, only_failed=not args.all, library=args.library, limit=args.limit)

    parts = [render_overview(merged, experiment_dir)]
    scope = "all problems" if args.all else "failed problems"
    if args.library:
        scope += f", library={args.library}"
    parts.append(f"\n---\n\n# Details ({len(selected)} shown — {scope})\n")
    for pid, record in selected:
        parts.append(render_problem(pid, record, max_chars))
        parts.append("\n---\n")

    out_path = args.out or os.path.join(experiment_dir, "report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"wrote {out_path}  ({len(selected)} problems)")

    if args.json:
        json_path = os.path.join(experiment_dir, "report.json")
        payload = [
            {"problem_id": pid, **record}
            for pid, record in sorted(merged.items(), key=lambda kv: str(kv[0]))
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
