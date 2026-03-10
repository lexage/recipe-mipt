from abc import ABC, abstractmethod
from typing import (
    List,
    Sequence,
    Tuple,
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
        
        documnets = "## Documents:\n"
        inter_steps = ""
        
        for chunk in chunks:
            if chunk.id == "corag_intermediate_steps":
                inter_steps = chunk.text
            else:
                documnets+=f"""### Doc {chunk.id}:\n{chunk.text.strip()}\n\n"""

        return documnets + "\n" + inter_steps


class InstructRAGContextAssembler(ContextAssembler):
    def assemble(self, data: Tuple[List[Text], List[Chunk]]):
        rationalities, chunks = data
        context = ""
        for i, chunk in enumerate(chunks):
            rationality = rationalities[i] if i < len(rationalities) else ""
            context += f"[CHUNK {chunk.id} | doc={chunk.doc_id}]\n"
            if rationality:
                context += f"Rationality: {rationality}\n"
            context += f"{chunk.text}\n\n"

        return context.strip()
