from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    List,
    Sequence,
)

from src.agent_constructor.core import Block, Chunk, Text

class Retriever(Block):
    """Retrieve relevant chunks for a query/context."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        raise NotImplementedError


class Reranker(Block):
    """Optionally re-rank retrieved chunks."""

    @abstractmethod
    def rank(self, query: str, candidates: Sequence[Chunk]) -> List[Chunk]:
        raise NotImplementedError


class ContextAssembler(Block):
    """Assemble final context (string) from chosen chunks + optional structured pieces.

    Different strategies: RAG (concatenate + source markers), iCL (few shot selection),
    Reasoning (chain-of-thought augmented context), or hybrid.
    """

    @abstractmethod
    def assemble(self, query: str, chunks: Sequence[Chunk]) -> Text:
        raise NotImplementedError

# ---------- Simple ContextAssembler implementation ----------
class SimpleContextAssembler(ContextAssembler):
    def assemble(self, query: str, chunks: Sequence[Chunk]) -> Text:
        # naive concatenation with headers
        parts = [f"[CHUNK {c.id} | doc={c.doc_id}]\n{c.text}" for c in chunks]
        return "\n\n".join(parts)
