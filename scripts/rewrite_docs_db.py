"""LLM-rewrite the documentation for the 'doc rewrite' ablation axis.

Copies a docs DB but replaces every document.content with a compact structured
rewrite produced by DocRewriter (src/agents/generation). Sections/examples/
libraries are copied unchanged. Needs the vLLM server (DocRewriter url).

This is a HEAVY pass: one LLM call per document (~18k for the full DB, ~17k for
the reference DB). It is resumable — already-rewritten docs (present in the
destination) are skipped, so you can Ctrl-C and re-run.

Usage:
  python scripts/rewrite_docs_db.py                 # full DB  -> ..._rewritten.db
  python scripts/rewrite_docs_db.py --src data/docs_database_examples_apiref.db \\
      --dst data/docs_database_examples_apiref_rewritten.db   # reference variant
"""
import argparse
import os
import sqlite3

from src.agents.generation.generation_agents import DocRewriter

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="C:/Projects/docs_database_examples.db")
ap.add_argument("--dst", default="C:/Projects/recipe-mipt/data/docs_database_examples_rewritten.db")
ap.add_argument("--url", default="http://localhost:7215/v1")
ap.add_argument("--max-tokens", type=int, default=768)
args = ap.parse_args()

os.makedirs(os.path.dirname(args.dst), exist_ok=True)
src = sqlite3.connect(args.src)
dst = sqlite3.connect(args.dst)

# recreate schema on first run
existing = {r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if "documents" not in existing:
    for (sql,) in src.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
    ):
        dst.execute(sql)
    dst.commit()
    dst.execute("ATTACH DATABASE ? AS s", (args.src,))
    dst.execute("INSERT INTO libraries SELECT * FROM s.libraries")
    dst.execute("INSERT INTO sections SELECT * FROM s.sections")
    dst.execute("INSERT INTO examples SELECT * FROM s.examples")
    dst.commit()
    dst.execute("DETACH DATABASE s")

done = {r[0] for r in dst.execute("SELECT id FROM documents")}
rows = src.execute("SELECT id, section_id, name, content, created_at FROM documents").fetchall()
todo = [r for r in rows if r[0] not in done]
print(f"rewriting {len(todo)} / {len(rows)} documents ({len(done)} already done) -> {args.dst}")

rewriter = DocRewriter(url=args.url, max_tokens=args.max_tokens)

for i, (doc_id, section_id, name, content, created_at) in enumerate(todo, 1):
    try:
        new_content = rewriter.run(content or "")
    except Exception as e:  # noqa: BLE001 - keep going, log the failure, keep raw
        print(f"  [warn] doc {doc_id}: rewrite failed ({e!r}); keeping raw")
        new_content = content
    dst.execute(
        "INSERT INTO documents (id, section_id, name, content, length, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, section_id, name, new_content, len(new_content or ""), created_at),
    )
    if i % 50 == 0:
        dst.commit()
        print(f"  {i}/{len(todo)}")

dst.commit()
print(f"DONE: {dst.execute('SELECT COUNT(*) FROM documents').fetchone()[0]} documents in {args.dst}")
src.close()
dst.close()
