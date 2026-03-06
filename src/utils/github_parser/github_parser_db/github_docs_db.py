"""
Реализация IDB для GitHub парсера.
Наследуется от абстрактного IDB и использует SQLite адаптер.
"""

from typing import Iterable, List, Optional

from src.agent_constructor.core import Chunk, Document, Text
from src.utils.adapters import ChromaDocsAdapter
from .base_idb import IDB
from .github_sqlite_adapter import GitHubSQLiteAdapter
from ..config import ExtractedExample


class GitHubDocsDB(IDB):
    """
    Реализация IDB для хранения примеров из GitHub парсера.
    Использует SQLite для хранения и векторный поиск через Chroma.
    """
    
    def __init__(
        self,
        embedder,
        db_path: str = "data/github_examples.db",
        vector_db_path: str = "data/github_vector_db",
        collection_name: str = "github_examples"
    ):
        self.adapter = GitHubSQLiteAdapter(db_path)
        self.vdb_adapter = ChromaDocsAdapter(
            embedder=embedder,
            collection_name=collection_name,
            path_to_db=vector_db_path
        )
        self._chunks_cache: dict = {}
    
    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Добавляет чанки в векторную БД."""
        chunks_list = list(chunks)
        if chunks_list:
            self.vdb_adapter.add(chunks_list)
            for chunk in chunks_list:
                self._chunks_cache[chunk.id] = chunk
    
    def query(self, query_text: str, top_k: int = 10) -> List[Chunk]:
        """Векторный поиск."""
        results = self.vdb_adapter.search(queries=[query_text], top_k=top_k)
        return results[0] if results else []
    
    def all_chunks(self) -> List[Chunk]:
        """Возвращает все чанки."""
        return self.vdb_adapter.get_chunks()
    
    def get_documents(self, ids: Optional[List[int]] = None) -> List[Document]:
        return []