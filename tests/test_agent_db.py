"""
Тест для проверки работы с БД через agent_constructor
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent_constructor.db import LocalDB
from src.agent_constructor.agent import Agent


class MockEmbedder:
    """Заглушка для эмбеддера (для теста)"""
    def __call__(self, text):
        return [0.1] * 384  # возвращаем фейковый эмбеддинг


def test_local_db():
    """Тестируем LocalDB"""
    
    
    # Путь к вашей существующей БД
    db_path = "data/docs_database.db"
    full_path = os.path.abspath(db_path)
    print(f" Путь к БД: {full_path}")
    
    # Проверяем, существует ли файл
    if not os.path.exists(full_path):
        print(f"БД не найдена: {full_path}")
        
        return
    
    try:
        # Создаём эмбеддер (заглушку для теста)
        embedder = MockEmbedder()
        
        # Подключаемся к существующей БД
        db = LocalDB(
            embedder=embedder,
            path_to_db=db_path,
            path_to_vector_db='data/docs_vector_database',
            collection_name='docs'
        )
        
       
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_local_db()