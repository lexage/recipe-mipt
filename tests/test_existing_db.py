"""
Тесты для проверки работы с уже существующей БД.
Полностью совместимо с архитектурой agent_constructor.
"""

import sys
import os
import sqlite3
import tempfile
import json
import time
import gc
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.github_parser import GitHubDocsDB, ExtractedExample
from src.agent_constructor.core import Chunk


def create_existing_db(db_path: str) -> None:
    """
    Создаёт существующую БД с тестовыми данными.
    Полностью соответствует структуре из adapters.py.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Таблица chunks (для поиска)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Таблица raw_examples (полные данные) - используем refs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_examples (
        id TEXT PRIMARY KEY,
        source_object_type TEXT NOT NULL,
        source_object_name TEXT NOT NULL,
        source_object_path TEXT NOT NULL,
        task_description TEXT,
        solution_code TEXT NOT NULL,
        metadata_source_code TEXT,
        refs TEXT,
        repository_name TEXT NOT NULL,
        chunk_index INTEGER DEFAULT 0,
        total_chunks INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Добавляем существующие данные
    existing_chunks = [
        {
            'id': 'old_chunk_1',
            'text': 'def old_function():\n    return "old data"',
            'metadata': json.dumps({'source': 'old_repo', 'type': 'function'})
        },
        {
            'id': 'old_chunk_2',
            'text': 'class OldClass:\n    pass',
            'metadata': json.dumps({'source': 'old_repo', 'type': 'class'})
        }
    ]
    
    for chunk in existing_chunks:
        cursor.execute("""
        INSERT OR IGNORE INTO chunks (id, text, metadata) VALUES (?, ?, ?)
        """, (chunk['id'], chunk['text'], chunk['metadata']))
    
    conn.commit()
    conn.close()
    time.sleep(0.2)  # Даём время на закрытие
    print(f"Создана существующая БД с {len(existing_chunks)} чанками")


def force_cleanup(file_path):
    """
    Принудительное удаление файла с несколькими попытками.
    Решает проблему 'файл занят другим процессом' в Windows.
    """
    if not os.path.exists(file_path):
        return
    
    # Принудительный сбор мусора
    gc.collect()
    time.sleep(0.3)
    
    for attempt in range(5):
        try:
            os.unlink(file_path)
            print(f"Файл удалён: {os.path.basename(file_path)}")
            return
        except PermissionError:
            wait_time = 0.5 * (attempt + 1)
            print(f"Попытка {attempt + 1}/5: файл занят, ждём {wait_time}с...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Ошибка при удалении: {e}")
            break
    
    print(f"Не удалось удалить {file_path} - будет удалён при перезагрузке")


def test_connect_to_existing_db():
    
    
    db_path = None
    try:
        # Создаём временный файл
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        # Создаём существующую БД
        create_existing_db(db_path)
        
        # Подключаемся к БД
        db = GitHubDocsDB(db_path)
        
        # Проверяем данные
        all_chunks = db.all_chunks()
        assert len(all_chunks) == 2, f"Ожидалось 2 чанка, получено {len(all_chunks)}"
        print(f"Найдено {len(all_chunks)} существующих чанков")
        
        # Проверяем ID
        chunk_ids = [chunk.id for chunk in all_chunks]
        assert 'old_chunk_1' in chunk_ids, "Старый чанк 1 не найден"
        assert 'old_chunk_2' in chunk_ids, "Старый чанк 2 не найден"
        print("Все существующие чанки доступны")
        
        # Явно удаляем объект и закрываем соединения
        db.adapter.close_all_connections()
        del db
        gc.collect()
        time.sleep(0.3)
        
    finally:
        if db_path and os.path.exists(db_path):
            force_cleanup(db_path)


def test_add_to_existing_db():
    """
    Тест 2: Добавление новых данных в существующую БД
    """
    
    
    db_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        create_existing_db(db_path)
        db = GitHubDocsDB(db_path)
        
        # Создаём новый пример
        new_example = ExtractedExample(
            source_object_type="function",
            source_object_name="new_function",
            source_object_path="new.py",
            task_description="New function added to existing DB",
            solution_code="def new_function():\n    return 'new data'",
            metadata_source_code="",
            references=""
        )
        
        # Добавляем
        added = db.add_extracted_examples([new_example], "new_repo")
        assert added == 1, f"Ожидалось добавление 1 примера, добавлено {added}"
        
        
        # Проверяем количество
        all_chunks = db.all_chunks()
        assert len(all_chunks) == 3, f"Ожидалось 3 чанка, получено {len(all_chunks)}"
        print(f" Всего чанков: {len(all_chunks)} (2 старых + 1 новый)")
        
        db.adapter.close_all_connections()
        del db
        gc.collect()
        
    finally:
        if db_path and os.path.exists(db_path):
            force_cleanup(db_path)


def test_duplicate_handling():
    """
    Тест 3: Проверка обработки дубликатов
    """
    
    
    db_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        create_existing_db(db_path)
        db = GitHubDocsDB(db_path)
        
        # Создаём тестовый пример
        test_example = ExtractedExample(
            source_object_type="function",
            source_object_name="test_duplicate",
            source_object_path="test.py",
            task_description="Test duplicate",
            solution_code="def test():\n    return 42",
            metadata_source_code="",
            references=""
        )
        
        # Первое добавление
        added1 = db.add_extracted_examples([test_example], "test_repo")
        assert added1 == 1, f"Первое добавление: ожидалось 1, получено {added1}"
        
        # Второе добавление (дубликат)
        added2 = db.add_extracted_examples([test_example], "test_repo")
        assert added2 == 0, f"Второе добавление: ожидалось 0, получено {added2}"
        
        
        
        db.adapter.close_all_connections()
        del db
        gc.collect()
        
    finally:
        if db_path and os.path.exists(db_path):
            force_cleanup(db_path)


def test_data_integrity():
    """
    Тест 4: Проверка целостности данных
    """
    
    
    db_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        create_existing_db(db_path)
        db = GitHubDocsDB(db_path)
        
        # Добавляем новый пример
        new_example = ExtractedExample(
            source_object_type="class",
            source_object_name="IntegrityTest",
            source_object_path="integrity.py",
            task_description="Test data integrity",
            solution_code="class IntegrityTest:\n    def test(self):\n        return True",
            metadata_source_code="",
            references="https://test.com"
        )
        
        db.add_extracted_examples([new_example], "integrity_repo")
        
        # Проверяем прямой запрос к SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Проверяем chunks
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunks_count = cursor.fetchone()[0]
        assert chunks_count == 3, f"В chunks ожидалось 3, найдено {chunks_count}"
        
        # Проверяем raw_examples
        cursor.execute("SELECT COUNT(*) FROM raw_examples")
        raw_count = cursor.fetchone()[0]
        assert raw_count == 1, f"В raw_examples ожидалось 1, найдено {raw_count}"
        
        conn.close()
        print("Целостность данных подтверждена")
        
        db.adapter.close_all_connections()
        del db
        gc.collect()
        
    finally:
        if db_path and os.path.exists(db_path):
            force_cleanup(db_path)


def test_query_existing_data():
    """
    Тест 5: Поиск по существующим данным
    """
    
    
    db_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        create_existing_db(db_path)
        db = GitHubDocsDB(db_path)
        
        # Тестовые запросы
        test_queries = [
            ("old", 2, "поиск 'old'"),
            ("function", 1, "поиск 'function'"),
            ("class", 1, "поиск 'class'"),
            ("nonexistent", 0, "поиск несуществующего")
        ]
        
        for query, expected, message in test_queries:
            results = db.query(query, top_k=5)
            assert len(results) == expected, f"{message}: ожидалось {expected}, найдено {len(results)}"
            print(f"{message}: {len(results)} результатов")
        
        db.adapter.close_all_connections()
        del db
        gc.collect()
        
    finally:
        if db_path and os.path.exists(db_path):
            force_cleanup(db_path)


def run_all_tests():
    """Запускает все тесты"""
    
    
    tests = [
        test_connect_to_existing_db,
        test_add_to_existing_db,
        test_duplicate_handling,
        test_data_integrity,
        test_query_existing_data
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print("\nТест успешен\n" + "-" * 70)
        except AssertionError as e:
            failed += 1
            print(f"\nОшибка в {test.__name__}: {e}\n" + "-" * 70)
        except Exception as e:
            failed += 1
            print(f"\nИсключение в {test.__name__}: {e}\n" + "-" * 70)
    
    
    print(f"ИТОГИ: {passed} тестов пройдено, {failed} тестов упало")
    
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())