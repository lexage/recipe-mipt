"""Diagnostic (B): what does a filter REMOVE / CHANGE?

Loads a few documents from a DB, chunks them exactly like the pipeline
(RecursiveChunker, max_chunk_size=1000), runs one filter, and prints a per-chunk
diff: chunks DROPPED entirely and chunks CHANGED (line-level unified diff). For
the *_llm filters this shows what the LLM turned the code into — i.e. whether
"repair" is corrupting working code.

RUN ON THE SERVER (filters need the embedder at :7216; *_llm filters also need
the LLM at :7215). Claude never runs it.

Examples (see prose in chat):
  python inspect_filter.py --filter f1v2     --db data/docs_database_examples.db            --n 3
  python inspect_filter.py --filter code_llm --db data/docs_database_examples_realistic.db  --n 3
  python inspect_filter.py --filter topic_llm --db data/docs_database_examples_realistic.db --n 40 --tail 40

Output: printed + saved to results/diag/filter_<kind>.txt
"""

import argparse
import difflib
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.agents.general import EmbeddingAgent
from src.utils.adapters import SQLiteDocsDBAdapter
from src.agent_constructor.chunkers import RecursiveChunker
from src.filtering.self_clean_v2 import SelfConsistencyCleanerV2
from src.filtering.code_aware import CodeAwareSelectCleaner, CodeAwareLLMCleaner
from src.filtering.topic_aware import TopicSelectCleaner, TopicLLMCleaner


EMBED_URL = "http://0.0.0.0:7216/v1"
EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
LLM_URL = "http://0.0.0.0:7215/v1"
LLM_MODEL = "Qwen/Qwen2.5-32B-Instruct"
TOPIC = ("Python data-science library usage and API documentation: numpy, "
         "pandas, scipy, scikit-learn, tensorflow, pytorch, matplotlib")


def build_filter(kind, embedder):
    if kind == "f1v2":
        return SelfConsistencyCleanerV2(embedder=embedder)
    if kind == "code_select":
        return CodeAwareSelectCleaner(embedder=embedder)
    if kind == "code_llm":
        return CodeAwareLLMCleaner(embedder=embedder, llm_url=LLM_URL,
                                   llm_model=LLM_MODEL, num_workers=4)
    if kind == "topic_select":
        return TopicSelectCleaner(embedder=embedder, benchmark_topic=TOPIC,
                                  llm_url=LLM_URL, llm_model=LLM_MODEL, num_workers=4)
    if kind == "topic_llm":
        return TopicLLMCleaner(embedder=embedder, benchmark_topic=TOPIC,
                               llm_url=LLM_URL, llm_model=LLM_MODEL, num_workers=4)
    raise SystemExit(f"unknown --filter {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True,
                    choices=["f1v2", "code_select", "code_llm", "topic_select", "topic_llm"])
    ap.add_argument("--db", default="data/docs_database_examples.db")
    ap.add_argument("--n", type=int, default=3, help="docs from the HEAD (on-topic)")
    ap.add_argument("--tail", type=int, default=0, help="docs from the TAIL (off-topic junk)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"results/diag/filter_{args.filter}.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    sqlite = SQLiteDocsDBAdapter(path_to_db=args.db)
    all_docs = sqlite.get_docs()
    picked = all_docs[: args.n]
    if args.tail:
        picked += all_docs[-args.tail:]
    print(f"loaded {len(all_docs)} docs; inspecting {len(picked)} "
          f"(head {args.n} + tail {args.tail}) from {args.db}")

    chunker = RecursiveChunker(max_chunk_size=1000)
    chunks = []
    for d in picked:
        chunks.extend(chunker.chunk(d))
    before = {c.id: (c.doc_id, c.text or "") for c in chunks}

    embedder = EmbeddingAgent(url=EMBED_URL, model_name=EMBED_MODEL)
    flt = build_filter(args.filter, embedder)
    kept = flt.apply(list(chunks))
    after = {c.id: (c.text or "") for c in kept}

    dropped = changed = unchanged = 0
    lines = [f"FILTER={args.filter}  DB={args.db}  chunks_in={len(before)} chunks_out={len(after)}", ""]
    for cid, (doc_id, btext) in before.items():
        if cid not in after:
            dropped += 1
            lines += ["=" * 90, f"DROPPED chunk id={cid} doc_id={doc_id}",
                      btext.strip()[:1500], ""]
            continue
        atext = after[cid]
        if atext == btext:
            unchanged += 1
            continue
        changed += 1
        diff = difflib.unified_diff(
            btext.splitlines(), atext.splitlines(),
            fromfile=f"{cid}:before", tofile=f"{cid}:after", lineterm="")
        lines += ["=" * 90, f"CHANGED chunk id={cid} doc_id={doc_id}"]
        lines += list(diff)
        lines += [""]

    summary = (f"\nSUMMARY  dropped={dropped}  changed={changed}  "
               f"unchanged={unchanged}  (of {len(before)} chunks)")
    lines.append(summary)
    report = "\n".join(lines)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
