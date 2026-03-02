from abc import ABC, abstractmethod
from typing import (
    List,
    Sequence,
    Dict,
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
    def assemble(self, chunks: Sequence[Chunk]) -> Text:
        # naive concatenation with headers
        parts = [f"[CHUNK {c.id} | doc={c.doc_id}]\n{c.text}" for c in chunks]
        return "\n\n".join(parts)


class CoRAGContextAssembler(ContextAssembler):
    def assemble(self, chunks: List[Chunk]):
        
        documnets = "##Document\n"
        inter_steps = ""
        
        for chunk in chunks:
            if chunk.id == "corag_intermediate_steps":
                inter_steps = chunk.text
            else:
                documnets+=f"""Doc {chunk.id}: {chunk.text.strip()}\n\n"""

        return documnets + "\n" + inter_steps


class InstructRAGContextAssembler(ContextAssembler):
    def assemble(self, data: List[Chunk]):
        
        original_chunks: Dict[str: Chunk] = {}
        rationalities_chunks: List[Chunk] = []

        for chunk in data:
            if chunk.metadata.get("source", None) == "instruct_rag":
                rationalities_chunks.append(chunk)
            else:
                original_chunks[chunk.id] = chunk
        
        context = ""

        for idx, chunk in enumerate(rationalities_chunks):
            context+=f"# Document [{idx}]:\n{original_chunks[chunk.doc_id].text}\n"
            context+=f"# Rationale [{idx}]:\n{chunk.text}\n\n"

        return context.strip()
