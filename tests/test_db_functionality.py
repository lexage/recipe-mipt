"""
Тесты для проверки функциональности БД через LocalDB
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent_constructor.db import LocalDB
from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Chunk, Document


class MockEmbedder:
    """Заглушка для эмбеддера"""
    def __call__(self, text):
        return [0.1] * 384


def test_db_connection():
    """Тест 1: Проверка подключения"""
    
    print("ТЕСТ 1: Подключение к БД")
    
    
    db_path = "data/docs_database.db"
    embedder = MockEmbedder()
    
    db = LocalDB(
        embedder=embedder,
        path_to_db=db_path,
        path_to_vector_db='data/docs_vector_database',
        collection_name='docs'
    )
    
    print("LocalDB создан")
    print(f"Тип: {type(db).__name__}")
    return db


def test_query(db):
    """Тест 2: Проверка поиска"""
    
    print("ТЕСТ 2: Поиск в БД")
    
    
    try:
        results = db.query("accuracy", top_k=3)
        print(f"Поиск выполнен, найдено: {len(results)} результатов")
        for i, chunk in enumerate(results):
            print(f"   {i+1}. {chunk.id}")
    except Exception as e:
        print(f"Поиск не работает (возможно БД пуста): {e}")


def test_get_documents(db):
    """Тест 3: Получение документов"""
   
    print("ТЕСТ 3: Получение документов")
    
    
    try:
        docs = db.get_documents()
        print(f"Получено документов: {len(docs)}")
    except Exception as e:
        print(f"Ошибка получения документов: {e}")


def test_add_chunks(db):
    """Тест 4: Добавление тестовых данных"""
    
    print("ТЕСТ 4: Добавление тестовых данных")
    
    
    # Создаём тестовый чанк с правильными параметрами
    test_chunk = Chunk(
        id="test_123",
        doc_id="doc_test_123",  
        text="def test_function():\n    return 'hello world'",
        metadata={"source": "test", "type": "function"}
    )
    
    try:
        db.add_chunks([test_chunk])
        print("Тестовые данные добавлены")
    except Exception as e:
        print(f"Ошибка добавления: {e}")
    



def run_all_tests():
    """Запуск всех тестов"""
    
    
    # Тест 1: Подключение
    db = test_db_connection()
    
    # Тест 2: Поиск
    test_query(db)
    
    # Тест 3: Документы
    test_get_documents(db)
    
    # Тест 4: Добавление
    test_add_chunks(db)
    
    
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    


if __name__ == "__main__":
    run_all_tests()