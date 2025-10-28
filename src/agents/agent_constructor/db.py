from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Dict,
    Iterable,
    List,
    Tuple,
)

from src.agents.agent_constructor.core import Chunk, Document, Text
from src.agents.agent_constructor.chunkers import Chunker
from src.agents.agent_constructor.agent import Agent
from src.utils.adapters import SQLiteDocsDBAdapter, ChromaDocsAdapter
from src.utils.wrappers import EmbeddingFunctionWrapper


class IDB(ABC):
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


class LocalDB(IDB):
    def __init__(
            self, 
            chunker: Chunker, 
            embedding_model: Agent,
            path_to_db: str = 'data/docs_database.db', 
            path_to_vector_db: str = 'data/docs_vector_database', 
            collection_name: str = 'docs',
            ):

        self.doc_data_base = SQLiteDocsDBAdapter(
            path_to_db=path_to_db
            )

        self.vector_data_base = ChromaDocsAdapter(
            collection_name=collection_name,
            path_to_db=path_to_vector_db,
            embedding_function=EmbeddingFunctionWrapper(embedding_model)
            )
        
        documents = self.doc_data_base.get_docs()
        
        self.chunks = {
            chunk.id: chunk for chunk in self._get_chunks(
                chunker=chunker, 
                documents=documents
                )
            }

        if not self.vector_data_base.populated:
            text_data = [chunk.text for chunk in self.all_chunks()]
            embedding_model.fit(text_data)
            self.vector_data_base.populate(self.all_chunks())

    def _get_chunks(self, chunker: Chunker, documents: List[Document]):
        chunks = []
        for document in documents:
            doc_chunks = chunker.chunk(document)
            chunks.extend(doc_chunks)
        return chunks
    
    def query(self, queries: Text, top_k: int) -> List[Chunk]:
        chunk_ids = self.vector_data_base.search(queries=[queries], top_k=top_k)          
        return [self.chunks[chunk_id] for chunk_id in chunk_ids]
    
    def all_chunks(self) -> List[Chunk]:
        return list(self.chunks.values())
    
    def add_chunks(self, chunks):
        return super().add_chunks(chunks)
