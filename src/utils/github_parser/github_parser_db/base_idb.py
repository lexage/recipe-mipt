"""
Базовый абстрактный класс IDB для работы с базой данных.
По архитектуре agent_constructor.
"""

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from src.agent_constructor.core import Block, Chunk, Document


class IDB(Block):
    """Для хранения чанков и запросов к БД."""
    
    @abstractmethod
    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Добавление чанков в базу данных."""
        raise NotImplementedError
    
    @abstractmethod
    def query(self, query_text: str, top_k: int = 10) -> List[Chunk]:
        
        raise NotImplementedError
    
    @abstractmethod
    def all_chunks(self) -> List[Chunk]:
        """Испрользуем все чанки"""
        raise NotImplementedError
    
    @abstractmethod
    def get_documents(self, ids: Optional[List[int]] = None) -> List[Document]:
        """Получение документов по идентификаторам или по всему, если их нет."""
        raise NotImplementedError