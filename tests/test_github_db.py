"""
Тесты для GitHubDocsDB.
"""

import sys
import os
import tempfile
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.github_parser.github_parser_db import GitHubDocsDB, DS1000Wrapper
from src.utils.github_parser.config import ExtractedExample


def test_github_docs_db():
    """Тестируем GitHubDocsDB"""
    
    
    # Создаем временную БД
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Инициализация
        db = GitHubDocsDB(db_path)
        
        
        # Создаем тестовый пример
        test_example = ExtractedExample(
            source_object_type="function",
            source_object_name="test_function",
            source_object_path="test.py",
            task_description="This is a test function",
            solution_code="def test():\n    return 42",
            metadata_source_code="def test():\n    return 42",
            references="https://example.com"
        )
        
        # Добавляем пример
        added = db.add_extracted_examples([test_example], "test_repo")
        assert added == 1, f"Ожидалось 1, получено {added}"
        print(f"Добавлен пример: {added}")
        
        # Поиск
        results = db.query("test", top_k=5)
        assert len(results) > 0, "Поиск не дал результатов"
        print(f"Поиск 'test': найдено {len(results)} чанков")
        
        # Статистика
        stats = db.get_stats()
        print(f"Статистика: {stats}")
        
        # Тест DS1000Wrapper
        
        
        wrapper = DS1000Wrapper(db)
        wrapper.load_sample_data()
        wrapper.dummy_run("accuracy", top_k=2)
        
        print("\nВсе тесты пройдены!")
        
    finally:
        # Очистка
        os.unlink(db_path)


if __name__ == "__main__":
    test_github_docs_db()