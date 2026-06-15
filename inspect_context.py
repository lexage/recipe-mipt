"""Diagnostic (A): does the dirt actually reach the answer?

For a few DS1000 tasks, retrieve + assemble the context the solver would see,
on the CLEAN corpus vs the DIRTY corpus, and dump them side by side. Read-only
over the ALREADY-BUILT vector indexes in data/vdb_exp5/ — NO rebuild, NO
re-embedding of the corpus (only the few query strings are embedded).

RUN ON THE SERVER (needs the embedder at :7216). Claude never runs it.

  python inspect_context.py --n 4

Output: printed + saved to results/diag/context.txt
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.agents.general import EmbeddingAgent
from src.db.docs_db import LocalDB
from src.benchmarks.ds1000 import DatasetDS1000


EMBED_URL = "http://0.0.0.0:7216/v1"
EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
TOP_K = 5

CLEAN_SQLITE = "data/docs_database_examples.db"
CLEAN_VDB = "data/vdb_exp5/simple_example_pure_clean"
DIRTY_SQLITE = "data/docs_database_examples_realistic.db"
DIRTY_VDB = "data/vdb_exp5/simple_example_pure_dirty"


def make_db(embedder, sqlite_path, vdb_path):
    return LocalDB(
        embedder=embedder,
        path_to_db=sqlite_path,
        path_to_vector_db=vdb_path,
        collection_name="docs",
        return_full_docs=False,
        return_examples=True,
        merge_examples=False,
    )


def assemble(chunks):
    """Replicates CoRAGContextAssembler.assemble (no intermediate steps here)."""
    sections = [c.text.strip() for c in chunks if c.id != "corag_intermediate_steps"]
    if not sections:
        return ""
    return "# Relevant API Reference\n\n" + "\n\n---\n\n".join(sections)


def retrieve_block(db, prompt):
    try:
        chunks = db.query(query_text=prompt, top_k=TOP_K)
    except Exception as exc:                       # noqa: BLE001
        return f"<query failed: {exc}>", ""
    ctx = assemble(chunks)
    info = [f"    chunk doc_id={c.doc_id} score={(c.metadata.get('score') or 0):.4f} "
            f"len={len(c.text or '')}" for c in chunks]
    return ctx, "\n".join(info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/ds1000/ds1000.jsonl.gz")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="results/diag/context.txt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tasks = list(DatasetDS1000(args.dataset))[: args.n]
    embedder = EmbeddingAgent(url=EMBED_URL, model_name=EMBED_MODEL)

    # CLEAN first (Qdrant local holds an exclusive lock -> one DB open at a time).
    clean_blocks = []
    clean = make_db(embedder, CLEAN_SQLITE, CLEAN_VDB)
    for t in tasks:
        clean_blocks.append(retrieve_block(clean, t.prompt))
    clean.close()

    dirty_blocks = []
    dirty = make_db(embedder, DIRTY_SQLITE, DIRTY_VDB)
    for t in tasks:
        dirty_blocks.append(retrieve_block(dirty, t.prompt))
    dirty.close()

    lines = []
    for i, t in enumerate(tasks):
        pid = t.metadata.get("problem_id", i)
        lib = t.metadata.get("library", "?")
        c_ctx, c_info = clean_blocks[i]
        d_ctx, d_info = dirty_blocks[i]
        lines += [
            "=" * 100,
            f"TASK {pid}  library={lib}",
            "-" * 100,
            "PROMPT (tail 600 chars):",
            (t.prompt or "")[-600:],
            "-" * 100,
            "[CLEAN] retrieved chunks:",
            c_info,
            "[CLEAN] assembled context:",
            c_ctx,
            "-" * 100,
            "[DIRTY] retrieved chunks:",
            d_info,
            "[DIRTY] assembled context:",
            d_ctx,
            "",
        ]
    report = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
