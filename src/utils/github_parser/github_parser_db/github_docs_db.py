"""
Реализация IDB для GitHub парсера.
Наследуется от абстрактного IDB и использует SQLite адаптер.
"""

from typing import Iterable, List, Optional

from src.agent_constructor.core import Chunk, Document, Text

from .base_idb import IDB
from .github_sqlite_adapter import GitHubSQLiteAdapter
from ..config import ExtractedExample


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
        # TODO: реализовать, если нужна поддержка Document
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