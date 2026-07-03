"""Thread-local execution tracer for RAG pipelines.

The tracer collects per-task intermediate artifacts (generated queries, selected
APIs, retrieved chunks, the final prompt, the model answer, ...) so results can be
audited and a taxonomy of failures can be built afterwards.

Design goals
------------
* **Flexible / config-driven.** There is no fixed schema enforced across pipelines.
  Each component annotates the *active* trace with whatever it produced. Components
  that are not part of a given pipeline simply contribute nothing, so the captured
  trace automatically matches the pipeline configuration.
* **Non-invasive.** Every method is a no-op when tracing is disabled or when there
  is no active trace, so instrumentation calls are safe to sprinkle anywhere and do
  not change pipeline behaviour when tracing is off.
* **Thread-safe.** ``DS1000.eval`` runs one task per worker thread, so the active
  trace is stored in :class:`threading.local`. Completed traces are appended to a
  single JSONL file under a lock.

Typical wiring (done by the benchmark, not by user code)::

    Tracer.configure(path="results/exp/traces.jsonl")   # once, before workers start
    # inside each worker thread:
    Tracer.start(problem_id, prompt, metadata)
    ...  # components call Tracer.set / add / record_chunks / step
    Tracer.finish(answer)                                # writes one JSONL line
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Iterable, List, Optional


class Tracer:
    """Static, thread-local collector of pipeline execution traces."""

    _local = threading.local()
    _lock = threading.Lock()
    _path: Optional[str] = None
    _enabled: bool = False

    # ---- lifecycle ----------------------------------------------------------

    @classmethod
    def configure(cls, path: Optional[str], enabled: bool = True) -> None:
        """Enable tracing and set the JSONL output path (one line per task)."""
        cls._path = path
        cls._enabled = bool(enabled and path)

    @classmethod
    def disable(cls) -> None:
        cls._enabled = False
        cls._path = None

    @classmethod
    def enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def start(cls, problem_id: Any, prompt: Optional[str], metadata: Optional[dict] = None) -> None:
        """Begin a fresh trace for the current worker thread."""
        if not cls._enabled:
            return
        cls._local.trace = {
            "problem_id": problem_id,
            "metadata": dict(metadata or {}),
            "prompt": prompt,
            "queries": [],           # search queries actually issued to the retriever
            "apis": [],              # api-selector output (raw + normalized), if any
            "retrieved_chunks": [],  # provenance + text of everything retrieved
            "rationales": [],        # instruct-rag style rationales, if any
            "context": None,         # final assembled context string
            "system_prompt": None,   # solver system prompt (chat api only)
            "final_prompt": None,    # exact prompt handed to the solver
            "answer": None,          # raw model answer (pre-postprocessing)
            "steps": [],             # ordered, generic step log
        }

    @classmethod
    def _active(cls) -> Optional[Dict[str, Any]]:
        if not cls._enabled:
            return None
        return getattr(cls._local, "trace", None)

    @classmethod
    def finish(cls, answer: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Close the active trace, append it to the JSONL file, and return it."""
        trace = cls._active()
        if trace is None:
            return None
        if answer is not None:
            trace["answer"] = answer
        cls._local.trace = None

        if cls._path:
            line = json.dumps(trace, ensure_ascii=False, default=str)
            with cls._lock:
                with open(cls._path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        return trace

    # ---- annotation API (all no-ops without an active trace) ----------------

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        trace = cls._active()
        if trace is not None:
            trace[key] = value

    @classmethod
    def add(cls, key: str, value: Any) -> None:
        trace = cls._active()
        if trace is not None:
            trace.setdefault(key, []).append(value)

    @classmethod
    def extend(cls, key: str, values: Iterable[Any]) -> None:
        trace = cls._active()
        if trace is not None:
            trace.setdefault(key, []).extend(list(values))

    @classmethod
    def step(cls, component: str, action: str, **data: Any) -> None:
        """Append a generic, ordered step entry (useful for multi-stage pipelines)."""
        trace = cls._active()
        if trace is not None:
            trace["steps"].append({"component": component, "action": action, **data})

    @classmethod
    def record_chunks(cls, chunks: Iterable[Any], stage: str = "retrieved") -> None:
        """Capture provenance + text for a list of :class:`Chunk` objects.

        Reads attributes defensively so it works for any chunk-like object and any
        retriever. Rationale chunks (``metadata.source in {instruct_rag,
        api_instruct_rag}``) are additionally mirrored into ``rationales`` for
        convenient inspection.
        """
        trace = cls._active()
        if trace is None or not chunks:
            return
        for chunk in chunks:
            metadata = dict(getattr(chunk, "metadata", None) or {})
            entry = {
                "stage": stage,
                "id": str(getattr(chunk, "id", "")),
                "doc_id": str(getattr(chunk, "doc_id", "")),
                "source": metadata.get("source"),
                "retrieval_api": metadata.get("retrieval_api"),
                "doc_name": metadata.get("doc_name"),
                "library": metadata.get("library"),
                "score": metadata.get("score"),
                "text": getattr(chunk, "text", ""),
            }
            trace["retrieved_chunks"].append(entry)

            if metadata.get("source") in ("instruct_rag", "api_instruct_rag"):
                trace["rationales"].append(
                    {
                        "retrieval_api": metadata.get("retrieval_api"),
                        "doc_id": str(getattr(chunk, "doc_id", "")),
                        "text": getattr(chunk, "text", ""),
                    }
                )
