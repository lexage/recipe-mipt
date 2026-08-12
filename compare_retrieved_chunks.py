"""Compare retrieved_chunks.jsonl between two DS1000 runs.

For every task that FAILED in the first run (no aug) and PASSED in the
second run (aug), report what changed in the retrieved chunks: which
chunks were dropped, which were newly retrieved, and which stayed.

retrieved_chunks.jsonl is only written when run_ds1000.py is run with
--log-chunks; results.csv (pass/fail per problem_id) is expected next to
each retrieved_chunks.jsonl file.

Run (locally, after results synced):
  python compare_retrieved_chunks.py
  python compare_retrieved_chunks.py path/to/first.jsonl path/to/second.jsonl
"""

import argparse
import csv
import json
import os


def _latest_run_dir(config_name, save_path="results"):
    base = os.path.join(save_path, config_name)
    if not os.path.isdir(base):
        return None
    subs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    return os.path.join(base, subs[-1]) if subs else None


def _default_chunks_path(config_name):
    run_dir = _latest_run_dir(config_name)
    if run_dir is None:
        return os.path.join("results", config_name, "<run_dir>", "retrieved_chunks.jsonl")
    return os.path.join(run_dir, "retrieved_chunks.jsonl")


def load_scores(chunks_path):
    results_path = os.path.join(os.path.dirname(chunks_path), "results.csv")
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"results.csv not found next to {chunks_path}")
    scores = {}
    with open(results_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[int(row["problem_id"])] = int(row["score"])
    return scores


def load_chunks(chunks_path):
    if not os.path.isfile(chunks_path):
        raise FileNotFoundError(
            f"{chunks_path} not found. It is only written when run_ds1000.py "
            "is run with --log-chunks — rerun the config with that flag."
        )
    records = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[int(rec["problem_id"])] = rec
    return records


def describe_chunk(chunk):
    text = chunk.get("text", "").strip().replace("\n", " ")
    return (
        f"    [{chunk.get('id')}] doc_id={chunk.get('doc_id')} "
        f"library={chunk.get('library')} score={chunk.get('score')} "
        f"chars={chunk.get('chars')}\n"
        f"      {text[:300]!r}"
    )


def diff_chunks(before, after):
    before_by_id = {c["id"]: c for c in before.get("chunks", [])}
    after_by_id = {c["id"]: c for c in after.get("chunks", [])}
    removed = [c for cid, c in before_by_id.items() if cid not in after_by_id]
    added = [c for cid, c in after_by_id.items() if cid not in before_by_id]
    kept = [cid for cid in before_by_id if cid in after_by_id]

    lines = []
    lines.append(f"  n_chunks: {before.get('n_chunks')} -> {after.get('n_chunks')}")
    lines.append(f"  context_chars: {before.get('context_chars')} -> {after.get('context_chars')}")
    lines.append(f"  kept ({len(kept)}): {kept}")
    lines.append(f"  removed ({len(removed)}):")
    for c in removed:
        lines.append(describe_chunk(c))
    lines.append(f"  added ({len(added)}):")
    for c in added:
        lines.append(describe_chunk(c))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "first", nargs="?",
        default=_default_chunks_path("simple_example_pure_merge"),
        help="retrieved_chunks.jsonl of the baseline (without aug) run.",
    )
    ap.add_argument(
        "second", nargs="?",
        default=_default_chunks_path("simple_example_pure_merge_aug"),
        help="retrieved_chunks.jsonl of the aug run.",
    )
    ap.add_argument(
        "-o", "--output", default="chunk_diff_failed_to_ok.txt",
        help="Where to write the report.",
    )
    args = ap.parse_args()

    scores_before = load_scores(args.first)
    scores_after = load_scores(args.second)
    chunks_before = load_chunks(args.first)
    chunks_after = load_chunks(args.second)

    flipped = sorted(
        pid for pid, score in scores_before.items()
        if score == 0 and scores_after.get(pid) == 1
    )

    with open(args.output, "w", encoding="utf-8") as out:
        out.write(f"first (baseline):  {args.first}\n")
        out.write(f"second (aug):      {args.second}\n")
        out.write(f"failed -> passed:  {len(flipped)} tasks\n")
        out.write("=" * 80 + "\n\n")
        for pid in flipped:
            before = chunks_before.get(pid)
            after = chunks_after.get(pid)
            meta = after or before or {}
            out.write(
                f"problem_id={pid} library={meta.get('library')} "
                f"perturbation={meta.get('perturbation_type')}\n"
            )
            if before is None or after is None:
                out.write("  (missing chunk record for this problem_id in one of the files)\n\n")
                continue
            out.write(diff_chunks(before, after) + "\n\n")

    print(f"{len(flipped)} tasks flipped failed -> passed")
    print(f"report written to {args.output}")


if __name__ == "__main__":
    main()
