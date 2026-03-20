from abc import ABC, abstractmethod
from typing import (
    Dict,
    Iterable,
    List,
    Tuple,
    Optional,
)

from src.agent_constructor.core import Chunk, Document, Block
from typing import Iterable, List, Optional

class IDB(Block):
    """Minimal DB abstraction for storing and querying chunks."""

    @abstractmethod
    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, query_text: str, top_k: int = 10) -> List[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def all_chunks(self) -> List[Chunk]:
        raise NotImplementedError
    
    @abstractmethod
    def get_documents(ids: Optional[List[int]]) -> List[Document]:
        raise NotImplementedError
    
    
class InMemoryDB(IDB):
    def __init__(self):
        self._chunks: Dict[str, Chunk] = {}

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c

    def query(self, query_text: str, top_k: int = 10) -> List[Chunk]:
        # naive substring scoring: longest overlap
        scored: List[Tuple[int, Chunk]] = []
        q_tokens = set(query_text.split())
        for c in self._chunks.values():
            score = len(q_tokens.intersection(set(c.text.split())))
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:top_k]]

    def all_chunks(self) -> List[Chunk]:
        return list(self._chunks.values())
