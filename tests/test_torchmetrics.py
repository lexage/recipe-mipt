"""
Тест для проверки данных torchmetrics в БД
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent_constructor.db import LocalDB
from src.agent_constructor.agent import Agent


class MockEmbedder:
    """Заглушка для эмбеддера"""
    def __call__(self, text):
        return [0.1] * 384


def find_torchmetrics_db():
    """Ищет БД с torchmetrics"""
    possible_paths = [
        "data/github_examples.db",
        "data/code_examples.db",
        "data/torchmetrics.db",
        "data/docs_database.db",
    ]
    
    for path in possible_paths:
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            print(f"Найдена БД: {full_path}")
            return path
    
    print("БД с torchmetrics не найдена")
    return None


def test_torchmetrics_data():
    """Проверяем данные torchmetrics"""
    
    
    
    # Ищем БД
    db_path = find_torchmetrics_db()
    if not db_path:
        
        print("   python -m src.utils.github_parser")
        return
    
    try:
        # Подключаемся к БД
        embedder = MockEmbedder()
        db = LocalDB(
            embedder=embedder,
            path_to_db=db_path,
            path_to_vector_db='data/torchmetrics_vectors',
            collection_name='torchmetrics'
        )
        
        
        docs = db.get_documents()
        print(f"   Найдено документов: {len(docs)}")
        
        # Пробуем поиск
        
        results = db.query("accuracy", top_k=5)
        print(f"   Найдено результатов: {len(results)}")
        
        if len(results) > 0:
            
            for i, chunk in enumerate(results):
                print(f"\n--- Результат {i+1} ---")
                print(f"   ID: {chunk.id}")
                print(f"   Текст: {chunk.text[:150]}...")
                if chunk.metadata:
                    print(f"   Метаданные: {chunk.metadata}")
        
        # Статистика
        
        print(f"   Всего документов: {len(docs)}")
        print(f"   Результатов поиска: {len(results)}")
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_torchmetrics_data()