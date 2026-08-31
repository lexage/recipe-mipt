"""Build the derived corpora for experiment 16 (ablation of the manual augmentation).

The augmented DB adds 197 documents on top of the base corpus and nothing else:
same schema, same 18 063 original documents byte-for-byte, same 45 867 examples.
Every corpus below is therefore produced from the AUGMENTED file by deleting a
subset of those 197 rows, which keeps the untouched part bit-identical across
all variants — the only thing that varies between runs is which guides survive.

Two independent groupings are used (see the exp16 README):

  by function      guide_* (17)  |  migr_* (40)  |  howto_* (140)
  by test coupling T1 (~30)      |  T2 (~28)     |  T3 (rest)

T1 is recomputed here rather than hard-coded: a new document lands in T1 when it
contains a quoted string literal that appears nowhere in the ENTIRE base corpus
(documents + examples). Those literals are column names and phrasings the
generating agent could only have taken from the benchmark tasks it was shown.
T2 is a keyword rule over grader/checker-facing advice.

Usage (from the repo root):

    python scripts/make_exp16_dbs.py \
        --base data/docs_database_examples.db \
        --aug  data/docs_database_examples_aug.db \
        --out  data/exp16

    # inspect the classification without writing 9 copies of a 110 MB file
    python scripts/make_exp16_dbs.py --base ... --aug ... --out ... --dry-run
"""

import argparse
import json
import os
import re
import shutil
import sqlite3

from collections import Counter


# --- classification rules ---------------------------------------------------

# Quoted literals and df["..."] keys: the fingerprint of a task-specific detail.
LITERAL_RE = re.compile(r"""['"]([A-Za-z][\w \-\(\)%\.]{2,40})['"]""")
DF_KEY_RE = re.compile(r"""df\[['\"]([A-Za-z_][\w ()]*)['\"]\]""")

# Advice that describes how the answer is CHECKED rather than how the code works.
GRADER_RE = re.compile(
    r"\b(checker|grader"
    r"|hidden (test|DataFrame|array|value|input)"
    r"|expected (output|column|shape|object|new)"
    r"|the task (says|asks|shows|lists|names|gives|wants)"
    r"|byte by byte|letter-for-letter|exactly as|match the exact|mirror the names)\b",
    re.IGNORECASE,
)

FUNCTION_GROUPS = ("guide", "migr", "howto")


def connect_ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def base_corpus_text(base_db: str) -> str:
    """Every character of the original corpus, for the novel-literal test."""
    parts = []
    with connect_ro(base_db) as conn:
        for (content,) in conn.execute("SELECT content FROM documents"):
            parts.append(content or "")
        for (content,) in conn.execute("SELECT content FROM examples"):
            parts.append(content or "")
    return "\n".join(parts)


def added_documents(base_db: str, aug_db: str):
    """Rows present in aug but not in base, as (id, name, content)."""
    with connect_ro(base_db) as conn:
        base_ids = {row[0] for row in conn.execute("SELECT id FROM documents")}
    with connect_ro(aug_db) as conn:
        rows = [
            (doc_id, name, content)
            for doc_id, name, content in conn.execute(
                "SELECT id, name, content FROM documents"
            )
            if doc_id not in base_ids
        ]
    return sorted(rows)


def classify(rows, base_text: str):
    """Return {doc_id: {"name", "function", "coupling", "novel_literals"}}."""
    table = {}
    for doc_id, name, content in rows:
        function = name.split("_", 1)[0]
        if function not in FUNCTION_GROUPS:
            function = "other"

        candidates = set(LITERAL_RE.findall(content or ""))
        candidates |= set(DF_KEY_RE.findall(content or ""))
        novel = sorted(literal for literal in candidates if literal not in base_text)

        if novel:
            coupling = "T1"
        elif GRADER_RE.search(content or ""):
            coupling = "T2"
        else:
            coupling = "T3"

        table[doc_id] = {
            "name": name,
            "function": function,
            "coupling": coupling,
            "novel_literals": novel,
        }
    return table


# --- corpus construction ----------------------------------------------------

def build(aug_db: str, out_path: str, drop_ids, vacuum: bool = True) -> int:
    """Copy the augmented DB and delete `drop_ids` from documents."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    shutil.copyfile(aug_db, out_path)

    conn = sqlite3.connect(out_path)
    try:
        drop_ids = list(drop_ids)
        # Chunked DELETE: SQLite caps a statement at 999 bound parameters.
        for start in range(0, len(drop_ids), 500):
            batch = drop_ids[start:start + 500]
            placeholders = ",".join("?" * len(batch))
            conn.execute(
                f"DELETE FROM documents WHERE id IN ({placeholders})", batch
            )
        conn.commit()

        # Drop sections/libraries left with no documents, so a variant that keeps
        # zero added docs is indistinguishable from the original base corpus.
        conn.execute(
            "DELETE FROM sections WHERE id NOT IN "
            "(SELECT DISTINCT section_id FROM documents)"
        )
        conn.execute(
            "DELETE FROM libraries WHERE id NOT IN "
            "(SELECT DISTINCT library_id FROM sections)"
        )
        conn.commit()

        remaining = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if vacuum:
            conn.execute("VACUUM")
    finally:
        conn.close()
    return remaining


def variants(table):
    """Corpus name -> set of added-doc ids to KEEP."""
    all_ids = set(table)
    by_function = {
        group: {i for i, meta in table.items() if meta["function"] == group}
        for group in FUNCTION_GROUPS
    }
    by_coupling = {
        tier: {i for i, meta in table.items() if meta["coupling"] == tier}
        for tier in ("T1", "T2", "T3")
    }

    plan = {
        # E0 — reference points
        "base": set(),
        "aug_full": all_ids,
        # E1 — one group at a time (base + group)
        "only_guide": by_function["guide"],
        "only_migr": by_function["migr"],
        "only_howto": by_function["howto"],
        # E1 — everything except one group (aug - group)
        "minus_guide": all_ids - by_function["guide"],
        "minus_migr": all_ids - by_function["migr"],
        "minus_howto": all_ids - by_function["howto"],
        # E2 — strip test-coupled documents
        "minus_t1": all_ids - by_coupling["T1"],
        "minus_t1_t2": all_ids - by_coupling["T1"] - by_coupling["T2"],
    }
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="data/docs_database_examples.db")
    parser.add_argument("--aug", default="data/docs_database_examples_aug.db")
    parser.add_argument("--out", default="data/exp16")
    parser.add_argument("--only", nargs="*",
                        help="Build only these variants (default: all).")
    parser.add_argument("--no-vacuum", action="store_true",
                        help="Skip VACUUM — faster, but every file stays full size.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the classification and exit.")
    args = parser.parse_args()

    print(f"- reading base corpus: {args.base}")
    base_text = base_corpus_text(args.base)
    print(f"  {len(base_text):,} chars indexed for the novel-literal test")

    rows = added_documents(args.base, args.aug)
    print(f"- documents added by the augmentation: {len(rows)}")

    table = classify(rows, base_text)

    functions = Counter(meta["function"] for meta in table.values())
    couplings = Counter(meta["coupling"] for meta in table.values())
    print(f"  by function: {dict(functions)}")
    print(f"  by coupling: {dict(couplings)}")
    print("  cross-tab (function x coupling):")
    for group in FUNCTION_GROUPS + ("other",):
        if not functions.get(group):
            continue
        cells = Counter(
            meta["coupling"] for meta in table.values() if meta["function"] == group
        )
        print(f"    {group:<6} T1={cells.get('T1', 0):>3}  "
              f"T2={cells.get('T2', 0):>3}  T3={cells.get('T3', 0):>3}")

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_db": args.base,
                "aug_db": args.aug,
                "n_added": len(table),
                "by_function": dict(functions),
                "by_coupling": dict(couplings),
                "documents": {str(k): v for k, v in sorted(table.items())},
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"- manifest -> {manifest_path}")

    plan = variants(table)
    if args.only:
        unknown = set(args.only) - set(plan)
        if unknown:
            raise SystemExit(f"unknown variant(s): {sorted(unknown)}")
        plan = {name: plan[name] for name in args.only}

    if args.dry_run:
        print("- dry run, no files written. Variants that would be built:")
        for name, keep in plan.items():
            print(f"    {name:<14} keeps {len(keep):>3} of {len(table)} added docs")
        return

    all_ids = set(table)
    for name, keep in plan.items():
        out_path = os.path.join(args.out, f"docs_{name}.db")
        drop = sorted(all_ids - keep)
        total = build(args.aug, out_path, drop, vacuum=not args.no_vacuum)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"- {name:<14} kept {len(keep):>3} added -> {total:>6} docs, "
              f"{size_mb:6.1f} MB  {out_path}")


if __name__ == "__main__":
    main()
