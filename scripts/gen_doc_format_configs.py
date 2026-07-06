"""Generate the 6 documentation-format ablation configs for CodeMMLU.

Fixed for every config (so only the doc FORMAT varies): Qwen2.5-32B-Instruct
solver (completions), SimpleRetriever + SimpleContextAssembler, Qwen3-Embedding-4B,
RecursiveChunker 1000, LengthFilter 20, top_k 5.

The 4 ablation axes map to:
  - DB filtering (full vs API reference only) -> path_to_db (full DB vs apiref DB)
  - Chunk content (theory vs +examples)       -> return_examples + merge_examples
  - Doc rewrite  (raw vs LLM-rewrite)          -> path_to_db (raw DB vs rewritten DB)
  - Granularity  (chunk vs whole document)     -> return_full_docs
"""
import os

CFG_DIR = "C:/Projects/recipe-mipt/pipeline_configs"

DB_FULL = "C:/Projects/docs_database_examples.db"
DB_APIREF = "C:/Projects/recipe-mipt/data/docs_database_examples_apiref.db"
DB_REWRITTEN = "C:/Projects/recipe-mipt/data/docs_database_examples_rewritten.db"
DB_APIREF_REWRITTEN = "C:/Projects/recipe-mipt/data/docs_database_examples_apiref_rewritten.db"

# name, comment, path_to_db, vdb, return_full_docs, return_examples, merge_examples
SPECS = [
    ("doc_0_baseline",  "full docs | theory only | raw | chunk (baseline)",
     DB_FULL, "vdb_doc_0_baseline", False, False, False),
    ("doc_1_apiref",    "API reference only | theory only | raw | chunk",
     DB_APIREF, "vdb_doc_1_apiref", False, False, False),
    ("doc_2_examples",  "full docs | theory + examples | raw | chunk",
     DB_FULL, "vdb_doc_2_examples", False, True, True),
    ("doc_3_rewrite",   "full docs | theory only | LLM-rewrite | chunk  (needs rewritten DB)",
     DB_REWRITTEN, "vdb_doc_3_rewrite", False, False, False),
    ("doc_4_fulldoc",   "full docs | theory only | raw | whole document",
     DB_FULL, "vdb_doc_4_fulldoc", True, False, False),
    ("doc_5_allon",     "API reference only | theory + examples | LLM-rewrite | whole document  (needs apiref+rewritten DB)",
     DB_APIREF_REWRITTEN, "vdb_doc_5_allon", True, True, True),
]

TEMPLATE = """# Doc-format ablation: {comment}
type: SIMPLE

params:
  top_k: 5

components:

  embedder:
    type: EMBEDDING_AGENT
    params:
      name: embedder
      url: "http://localhost:7216/v1"
      model_name: "Qwen/Qwen3-Embedding-4B"

  data_base:
    type: LOCAL_DB
    params:
      path_to_db: "{db}"
      path_to_vector_db: "data/{vdb}"
      collection_name: "docs"
      return_full_docs: {full}
      return_examples: {ex}
      merge_examples: {merge}
      hnsw_space: "cosine"
      hnsw_m: 16
      hnsw_construction_ef: 400
      hnsw_search_ef: 200
      batch_size: 16
      similarity_threshold: 0.1
      search_filter:
        source: "documents"

  chunker:
    type: RECURSIVE_CHUNKER
    params:
      name: chunker
      max_chunk_size: 1000

  filter:
    type: LENGTH_FILTER
    params:
      name: filter
      min_len: 20

  context_assembler:
    type: SIMPLE_CONTEXT_ASSEMBLER
    params:
      name: simple_assembler

  retriever:
    type: SIMPLE_RETRIEVER
    params:
      name: simple_retriever

  agent:
    type: CODEMMLU_SOLVER_AGENT
    params:
      name: codemmlu_solver
      url: "http://localhost:7215/v1"
      model_name: "Qwen/Qwen2.5-32B-Instruct"
      api: completions
      context_after_task: False
      temperature: 0.2
      top_p: 0.95
      max_tokens: 1024
      stop_tokens: []
"""

os.makedirs(CFG_DIR, exist_ok=True)
for name, comment, db, vdb, full, ex, merge in SPECS:
    text = TEMPLATE.format(comment=comment, db=db, vdb=vdb, full=full, ex=ex, merge=merge)
    path = os.path.join(CFG_DIR, f"ablation_{name}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {path}")
