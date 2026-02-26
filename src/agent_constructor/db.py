from abc import ABC, abstractmethod
from typing import (
    Dict,
    Iterable,
    List,
    Tuple,
    Optional,
)

from src.agent_constructor.core import Chunk, Document, Text, Block
from src.agent_constructor.chunkers import Chunker
from src.agent_constructor.agent import Agent
from src.utils.adapters import SQLiteDocsDBAdapter, ChromaDocsAdapter
from typing import Iterable, List, Optional
from .github_sqlite_adapter import GitHubSQLiteAdapter
from ..config import ExtractedExample

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


class LocalDB(IDB):
    def __init__(
            self,
            embedder: Agent,
            path_to_db: str = 'data/docs_database.db', 
            path_to_vector_db: str = 'data/docs_vector_database', 
            collection_name: str = 'docs',
            ):

        self.sqlite_adapter = SQLiteDocsDBAdapter(
            path_to_db=path_to_db
            )
        
        self.vdb_adapter = ChromaDocsAdapter(
            embedder=embedder,
            collection_name=collection_name,
            path_to_db=path_to_vector_db
        )
    
    def get_documents(self, ids: List[int] = None) -> List[Document]:
        if not ids:
            documents = self.sqlite_adapter.get_docs()
            documents.extend(self.sqlite_adapter.get_examples())
        else:
            documents = self.sqlite_adapter.get_docs(ids)
        return documents

    def query(self, query_text: Text, top_k: int = 10) -> List[Chunk]:
        chunks = self.vdb_adapter.search(queries=[query_text], top_k=top_k)[0]      
        return chunks
    
    def all_chunks(self) -> List[Chunk]:
        return self.vdb_adapter.get_chunks()
    
    def add_chunks(self, chunks: List[Chunk]):
        self.vdb_adapter.add(chunks)


class LocalRaptorDB(IDB):
    def __init__(
            self, 
            chunker: Chunker, 
            path_to_db: str = 'data/docs_database.db', 
            ):

        self.doc_data_base = SQLiteDocsDBAdapter(
            path_to_db=path_to_db
            )
        
        documents = self.doc_data_base.get_docs()
        
        self.chunks = {
            chunk.id: chunk for chunk in self._get_chunks(
                chunker=chunker, 
                documents=documents
                )
            }

    def _get_chunks(self, chunker: Chunker, documents: List[Document]):
        chunks = []
        for document in documents:
            doc_chunks = chunker.chunk(document)
            chunks.extend(doc_chunks)
        return chunks
    
    def query(self, queries: Text, top_k: int):
        pass

    def all_chunks(self):
        return list(self.chunks.values())
    
    def add_chunks(self, chunks):
        pass
"""
Реализация IDB для GitHub парсера.
Наследуется от абстрактного IDB и использует SQLite адаптер.
"""
class GitHubDocsDB(IDB):
    """
    Реализация IDB для хранения примеров из GitHub парсера.
    Использует SQLite для хранения и текстовый поиск.
    """
    
    def __init__(self, db_path: str = "data/github_examples.db"):
        self.adapter = GitHubSQLiteAdapter(db_path)
        self._chunks_cache: dict = {}
    
    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Добавляет чанки в БД."""
        for chunk in chunks:

            self._chunks_cache[chunk.id] = chunk
    
    def query(self, query_text: str, top_k: int = 10) -> List[Chunk]:
        """Текстовый поиск по чанкам."""
        return self.adapter.search(query_text, limit=top_k)
    
    def all_chunks(self) -> List[Chunk]:
        """Возвращает все чанки."""
        return self.adapter.get_all_chunks()
    
    def get_documents(self, ids: Optional[List[int]] = None) -> List[Document]:
        """
        Получает документы по ids.
        Каждый пример - это документ.
        """
        return []
    
    def add_extracted_examples(self, examples: List[ExtractedExample], repo_name: str) -> int:
        """
        Специфический метод для добавления результатов парсинга.
        Конвертирует ExtractedExample в чанк и сохраняет.
        """
        return self.adapter.add_examples_batch(examples, repo_name)
    
    def get_examples_by_repo(self, repo_name: str, limit: int = 100) -> List[dict]:
        """Получает примеры по репозиторию."""
        return self.adapter.get_examples_by_repo(repo_name, limit)
    
    def get_stats(self) -> dict:
        """Статистика по БД."""
        return self.adapter.get_stats()