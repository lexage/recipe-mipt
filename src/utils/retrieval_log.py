"""Per-task retrieval log — what the retriever actually gave each DS1000 task.

`bench.eval` drives ONE pipeline instance from several worker threads, so the
retrieved chunks have to be stashed per-thread: a plain attribute on the
pipeline would be overwritten by a concurrent task and silently attributed to
the wrong problem_id. Same reasoning (and same mechanism) as the thread-local
active tracker in `token_tracker`.

The pipeline records; `run_ds1000.py` calls `take()` right after
`pipeline.run()` returns — on the same thread — and writes the record out
keyed by problem_id.
"""

import threading

from typing import Dict, List

from src.agent_constructor.core import Chunk


_local = threading.local()


def record_chunks(chunks: List[Chunk]) -> None:
    """Remember the chunks retrieved for the task running on this thread."""
    _local.chunks = list(chunks)


def record_context(context: str) -> None:
    """Remember the assembled context string handed to the solver."""
    _local.context = context


def take() -> Dict:
    """Return this thread's record and clear it.

    Clearing matters: worker threads are reused across tasks, so a leftover
    value would be logged a second time for a task that retrieved nothing.
    """
    record = {
        "chunks": getattr(_local, "chunks", None) or [],
        "context": getattr(_local, "context", None),
    }
    _local.chunks = None
    _local.context = None
    return record
