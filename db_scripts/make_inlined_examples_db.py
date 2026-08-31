"""Materialise `<example_N>` placeholders into documents.content (experiment 6).

In the shipped corpus a document's code lives in a separate `examples` table and
the prose only holds a marker:

    ... the following call sorts in place: <example_3> ...

`LocalDB` indexes `documents` only, so the embedder never sees that code — it
embeds prose with holes in it. The substitution happens at QUERY time via
`merge_examples`, i.e. after retrieval has already picked the chunk. Documents
written by the augmentation agent carry their code inline instead, which makes
them structurally better retrievable than the original corpus regardless of what
they say. This script removes that asymmetry so the two can be compared fairly.

The replacement text matches `replace_examples_in_chunks` exactly (a newline
followed by the example body, several examples with the same order_id joined by
newlines), so an inlined document reads the same as a merged chunk.

Usage (from the repo root):

    python scripts/make_inlined_examples_db.py \
        --in  data/docs_database_examples.db \
        --out data/exp16/docs_base_inlined.db

    python scripts/make_inlined_examples_db.py \
        --in  data/exp16/docs_aug_full.db \
        --out data/exp16/docs_aug_inlined.db

`examples` is left untouched, so `return_examples` / `merge_examples` keep
working; placeholders are simply already resolved by the time they are read.
"""

import argparse
import os
import re
import shutil
import sqlite3

from collections import defaultdict


PLACEHOLDER_RE = re.compile(r"<example_(\d+)>")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--keep-unresolved", action="store_true",
                        help="Leave a placeholder as-is when no example matches "
                             "(default: same behaviour, flag is documentation).")
    parser.add_argument("--no-vacuum", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    shutil.copyfile(args.src, args.dst)

    conn = sqlite3.connect(args.dst)
    try:
        examples = defaultdict(list)
        for doc_id, order_id, content in conn.execute(
            "SELECT doc_id, order_id, content FROM examples ORDER BY id"
        ):
            examples[(doc_id, order_id)].append(content or "")
        merged = {key: "\n".join(values) for key, values in examples.items()}

        rows = list(conn.execute("SELECT id, content FROM documents"))

        updates = []
        stats = {"docs": len(rows), "touched": 0, "resolved": 0, "unresolved": 0}
        for doc_id, content in rows:
            if not content or "<example_" not in content:
                continue

            def substitute(match):
                key = (doc_id, int(match.group(1)))
                if key in merged:
                    stats["resolved"] += 1
                    return "\n" + merged[key]
                stats["unresolved"] += 1
                return match.group(0)

            new_content = PLACEHOLDER_RE.sub(substitute, content)
            if new_content != content:
                stats["touched"] += 1
                updates.append((new_content, len(new_content), doc_id))

        conn.executemany(
            "UPDATE documents SET content = ?, length = ? WHERE id = ?", updates
        )
        conn.commit()
        if not args.no_vacuum:
            conn.execute("VACUUM")
    finally:
        conn.close()

    size_mb = os.path.getsize(args.dst) / (1024 * 1024)
    print(f"- documents            : {stats['docs']}")
    print(f"- documents rewritten  : {stats['touched']}")
    print(f"- placeholders resolved: {stats['resolved']}")
    print(f"- placeholders left    : {stats['unresolved']} (no matching example row)")
    print(f"- written -> {args.dst}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
