"""Corpus build stage shared by the pipeline templates (exp17).

Splits corpus construction from vectorisation. The stage does, in order:

    load documents -> [document filter] -> [generation] -> [document filter]
                   -> [materialise to a new .db] -> hand documents to the chunker

Materialisation is the point of the module: the generated documents used to go
straight into the vector index and vanished with it, so a generated corpus could
never be inspected, diffed against the manual one, or re-run without paying the
generation cost again (43-57 minutes of LLM calls per exp15 config). Here the
source .db is copied under a new name, the generated documents are INSERTed into
it as ordinary rows with integer ids, and provenance goes into two extra tables
(``generation_meta``, ``generation_plans``). The resulting file is exactly the
corpus that gets indexed, so pointing ``path_to_db`` at it in a config without a
generator reproduces the run.

Two properties worth stating because they are easy to assume wrongly:

* materialisation happens after all DOCUMENT-level operations and before
  chunking, so a chunk-level filter is NOT represented in the file — a rerun
  from the file must keep the same chunk filter in the config;
* pruning of filtered-out corpus documents rewrites the ``documents`` table of
  the copy (and drops orphaned ``examples``), so the file never claims to hold
  documents that were not indexed.
"""

import hashlib
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agent_constructor.core import Document
from src.utils import DOCUMENT_SRC_DOCUMENTS

logger = logging.getLogger(__name__)

STAGE_PRE_GENERATION = "pre_generation"
STAGE_POST_GENERATION = "post_generation"
_STAGES = (STAGE_PRE_GENERATION, STAGE_POST_GENERATION)

ON_EXISTS_REBUILD = "rebuild"
ON_EXISTS_REUSE = "reuse"
ON_EXISTS_FAIL = "fail"
_ON_EXISTS = (ON_EXISTS_REBUILD, ON_EXISTS_REUSE, ON_EXISTS_FAIL)


@dataclass
class CorpusBuild:
    """Documents to index plus everything the runner should record."""

    documents: List[Document]
    document_filter_time: float = 0.0
    generation_time: float = 0.0
    materialize_time: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- helpers
def _source_db_path(data_base) -> str:
    adapter = getattr(data_base, "sqlite_adapter", None)
    return getattr(adapter, "path_to_db", "") if adapter is not None else ""


def _is_corpus_doc(doc: Document) -> bool:
    return str(doc.id).isdigit() and doc.source == DOCUMENT_SRC_DOCUMENTS


def _prepare_target(path: str, src: str, on_exists: str) -> bool:
    """Copy the source db to `path`. Returns True when generation must run."""
    if on_exists not in _ON_EXISTS:
        raise ValueError(f"on_exists must be one of {_ON_EXISTS}, got {on_exists!r}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        if on_exists == ON_EXISTS_FAIL:
            raise FileExistsError(
                f"materialized db already exists: {path} (on_exists='fail')")
        if on_exists == ON_EXISTS_REUSE:
            logger.warning("REUSING existing materialized db %s — generation and "
                           "document filtering are SKIPPED for this run", path)
            return False
        logger.warning("REBUILDING materialized db %s (deleting the old file)", path)
        os.remove(path)
    logger.info("materializing corpus: %s -> %s", src, path)
    shutil.copyfile(src, path)
    return True


def _ensure_meta_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generation_meta (
            doc_id      INTEGER PRIMARY KEY,
            doc_name    TEXT,
            method      TEXT,
            model       TEXT,
            endpoint    TEXT,
            seed        INTEGER,
            doc_class   TEXT,
            plan_id     TEXT,
            topic       TEXT,
            api_token   TEXT,
            prompt_sha1 TEXT,
            attempts    INTEGER,
            created_at  TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generation_plans (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            method     TEXT,
            library    TEXT,
            seed       INTEGER,
            status     TEXT,
            plan_id    TEXT,
            topic      TEXT,
            created_at TEXT,
            raw        TEXT
        )""")


def _section_id(conn: sqlite3.Connection, library: str, section: str) -> int:
    row = conn.execute("SELECT id FROM libraries WHERE name = ?", (library,)).fetchone()
    if row is None:
        cur = conn.execute("INSERT INTO libraries (name, description) VALUES (?, ?)",
                           (library, f"generated documents for {library}"))
        lib_id = cur.lastrowid
    else:
        lib_id = row[0]
    row = conn.execute(
        "SELECT id FROM sections WHERE library_id = ? AND name = ?",
        (lib_id, section)).fetchone()
    if row is not None:
        return row[0]
    cur = conn.execute(
        "INSERT INTO sections (library_id, name, path) VALUES (?, ?, ?)",
        (lib_id, section, f"{library}/{section}"))
    return cur.lastrowid


def _prune_corpus(conn: sqlite3.Connection, kept_ids: List[int],
                  prune_examples: bool) -> Dict[str, int]:
    """Delete corpus rows a document filter dropped, keeping the file honest."""
    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if len(kept_ids) == total:
        return {"pruned_documents": 0, "pruned_examples": 0}
    conn.execute("CREATE TEMP TABLE _kept (id INTEGER PRIMARY KEY)")
    conn.executemany("INSERT OR IGNORE INTO _kept (id) VALUES (?)",
                     [(i,) for i in kept_ids])
    dropped = conn.execute(
        "DELETE FROM documents WHERE id NOT IN (SELECT id FROM _kept)").rowcount
    dropped_ex = 0
    if prune_examples:
        dropped_ex = conn.execute(
            "DELETE FROM examples WHERE doc_id NOT IN (SELECT id FROM _kept)").rowcount
    conn.execute("DROP TABLE _kept")
    return {"pruned_documents": dropped, "pruned_examples": dropped_ex}


def _materialize(src_db: str, dst_db: str, corpus_docs: List[Document],
                 synth_docs: List[Document], generator, seed: Optional[int],
                 on_exists: str, prune_examples: bool) -> Dict[str, Any]:
    """Write the indexed corpus to its own .db and renumber generated docs."""
    if not _prepare_target(dst_db, src_db, on_exists):
        return {"materialized_db": dst_db, "reused": True}

    method = type(generator).__name__ if generator is not None else ""
    model = getattr(generator, "model_name", "") or ""
    endpoint = getattr(generator, "endpoint_url", "") or ""
    audit = getattr(generator, "audit_docs", {}) or {}
    plans = getattr(generator, "audit_plans", []) or []
    stamp = datetime.now().isoformat(timespec="seconds")

    stats: Dict[str, Any] = {"materialized_db": dst_db, "reused": False}
    conn = sqlite3.connect(dst_db)
    try:
        _ensure_meta_tables(conn)
        kept = [int(d.id) for d in corpus_docs if _is_corpus_doc(d)]
        stats.update(_prune_corpus(conn, kept, prune_examples))

        next_id = (conn.execute("SELECT COALESCE(MAX(id), 0) FROM documents")
                   .fetchone()[0]) + 1
        sections: Dict[str, int] = {}
        for doc in synth_docs:
            meta = dict(doc.metadata or {})
            library = meta.get("library") or "general"
            section = meta.get("section") or "generated"
            key = f"{library}/{section}"
            if key not in sections:
                sections[key] = _section_id(conn, library, section)
            doc_name = meta.get("doc_name") or f"gen_{next_id}"
            conn.execute(
                "INSERT INTO documents (id, section_id, name, content, length, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (next_id, sections[key], doc_name, doc.text, len(doc.text), stamp))
            record = audit.get(doc_name, {})
            conn.execute(
                "INSERT OR REPLACE INTO generation_meta (doc_id, doc_name, method, "
                "model, endpoint, seed, doc_class, plan_id, topic, api_token, "
                "prompt_sha1, attempts, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (next_id, doc_name, method, model, endpoint, seed,
                 record.get("doc_class", ""), record.get("plan_id", ""),
                 record.get("topic", ""), record.get("api_token", ""),
                 record.get("prompt_sha1", ""), record.get("attempts", 1), stamp))
            # The in-memory document must carry the same id, otherwise the vector
            # index and the .db disagree about what a doc_id means (exp15 shipped
            # string ids for generated docs and int ids for corpus ones).
            doc.id = str(next_id)
            meta["doc_id"] = next_id
            doc.metadata = meta
            next_id += 1

        for item in plans:
            conn.execute(
                "INSERT INTO generation_plans (method, library, seed, status, "
                "plan_id, topic, created_at, raw) VALUES (?,?,?,?,?,?,?,?)",
                (method, item.get("library", ""), seed, item.get("status", ""),
                 item.get("plan_id", ""), item.get("topic", ""), stamp,
                 item.get("raw", "")))
        conn.commit()
        stats["generated_rows"] = len(synth_docs)
        stats["plan_rows"] = len(plans)
        stats["documents_in_db"] = conn.execute(
            "SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        conn.close()
    logger.info("materialized corpus -> %s (%d documents, %d generated)",
                dst_db, stats.get("documents_in_db", -1), len(synth_docs))
    return stats


def _load_from_db(path: str) -> List[Document]:
    from src.utils.adapters.sqlite_adapter import SQLiteDocsDBAdapter
    return SQLiteDocsDBAdapter(path_to_db=path).get_docs()


def _prompt_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------------- stage
def build_corpus(
    data_base,
    generator=None,
    document_filter=None,
    document_filter_stage: str = STAGE_PRE_GENERATION,
    keep_source_docs: bool = True,
    materialize_db_path: str = "",
    on_exists: str = ON_EXISTS_REBUILD,
    generation_seed: Optional[int] = None,
) -> CorpusBuild:
    """Load, optionally filter and generate, optionally materialise.

    Every stage is optional and the four combinations of (filter, generator) all
    work; with neither, this is exactly the historical behaviour plus a load.
    """
    if document_filter_stage not in _STAGES:
        raise ValueError(
            f"document_filter_stage must be one of {_STAGES}, "
            f"got {document_filter_stage!r}")

    stats: Dict[str, Any] = {
        "document_filter_stage": document_filter_stage if document_filter else None,
        "generator": type(generator).__name__ if generator is not None else None,
        "generation_seed": generation_seed,
        "keep_source_docs": keep_source_docs,
    }

    # ---- reuse path: the materialized corpus already exists and is trusted ---
    if (materialize_db_path and on_exists == ON_EXISTS_REUSE
            and os.path.exists(materialize_db_path)):
        documents = _load_from_db(materialize_db_path)
        stats.update({"materialized_db": materialize_db_path, "reused": True,
                      "n_documents": len(documents)})
        logger.warning("corpus loaded from existing %s (%d documents); filter and "
                       "generator NOT run", materialize_db_path, len(documents))
        return CorpusBuild(documents=documents, stats=stats)

    documents = data_base.get_documents()
    stats["n_source_documents"] = len(documents)

    doc_filter_time = 0.0
    if document_filter and document_filter_stage == STAGE_PRE_GENERATION:
        t0 = time.time()
        documents = document_filter.apply(documents)
        doc_filter_time = time.time() - t0
        stats["n_after_document_filter"] = len(documents)

    synth_docs: List[Document] = []
    generation_time = 0.0
    if generator is not None:
        if generation_seed is not None and hasattr(generator, "seed"):
            generator.seed = int(generation_seed)
        t0 = time.time()
        synth_docs = generator.generate(documents=documents) or []
        generation_time = time.time() - t0
        stats["n_generated"] = len(synth_docs)
        stats["generation_stats"] = dict(getattr(generator, "stats", {}) or {})
        documents = (documents + synth_docs) if keep_source_docs else list(synth_docs)

    if document_filter and document_filter_stage == STAGE_POST_GENERATION:
        t0 = time.time()
        before = {id(d) for d in synth_docs}
        documents = document_filter.apply(documents)
        doc_filter_time = time.time() - t0
        stats["n_after_document_filter"] = len(documents)
        kept = {id(d) for d in documents}
        synth_docs = [d for d in synth_docs if id(d) in kept]
        stats["n_generated_after_filter"] = len(synth_docs)
        stats["n_generated_dropped_by_filter"] = len(before - kept)

    materialize_time = 0.0
    if materialize_db_path:
        src = _source_db_path(data_base)
        if not src:
            logger.warning("cannot materialize: data_base exposes no sqlite path")
        else:
            synth_ids = {id(d) for d in synth_docs}
            corpus_docs = [d for d in documents if id(d) not in synth_ids]
            non_document_sources = any(
                d.source != DOCUMENT_SRC_DOCUMENTS for d in corpus_docs)
            if non_document_sources:
                logger.warning("corpus contains non-'documents' sources — the "
                               "examples table is left untouched in the copy")
            t0 = time.time()
            stats.update(_materialize(
                src_db=src, dst_db=materialize_db_path, corpus_docs=corpus_docs,
                synth_docs=synth_docs, generator=generator, seed=generation_seed,
                on_exists=on_exists, prune_examples=not non_document_sources))
            materialize_time = time.time() - t0

    stats["n_documents"] = len(documents)
    return CorpusBuild(
        documents=documents,
        document_filter_time=doc_filter_time,
        generation_time=generation_time,
        materialize_time=materialize_time,
        stats=stats,
    )
