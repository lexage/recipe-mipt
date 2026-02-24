"""
Тест для проверки IDB
"""

import sys
import os
import logging
import tempfile
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.github_parser.idb import IDB, DS1000Wrapper
from src.utils.github_parser.config import ExtractedExample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_example(name: str = "test_function") -> ExtractedExample:
    """Создание тестового примера"""
    return ExtractedExample(
        source_object_type="function",
        source_object_name=name,
        source_object_path="test.py",
        task_description="This is a test function for accuracy metric",
        solution_code="def test():\n    return 42",
        metadata_source_code="def test():\n    return 42",
        references="https://arxiv.org/abs/1234.56789"
    )


def test_idb_basic():
    """Тест 1: Базовые операции"""
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Базовые операции IDB")
    print("=" * 60)
    
    # Создаем временную БД
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Инициализация
        db = IDB(db_path)
        print("IDB инициализирован")
        
        # Добавление примера
        example = create_test_example()
        result = db.add_extracted_examples_batch([example], "test_repo")
        assert result == 1, f"Ожидалось 1, получено {result}"
        print(f"Пример добавлен: {result}")
        
        # Поиск
        results = db.search("accuracy")
        assert len(results) > 0, "Поиск не дал результатов"
        print(f"Поиск 'accuracy': найдено {len(results)}")
        
        # Статистика
        stats = db.get_stats()
        assert stats['total_examples'] > 0, "Статистика пуста"
        print(f"Статистика: {stats}")
        
        print("\nВсе базовые тесты пройдены!")
        
    finally:
        # Очистка
        os.unlink(db_path)


def test_idb_references():
    """Тест 2: Сохранение ссылок"""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Сохранение ссылок")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        db = IDB(db_path)
        
        # Пример со ссылкой
        example = create_test_example()
        db.add_extracted_examples_batch([example], "test_repo")
        
        # Проверяем через прямой запрос к БД
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM refs")
            ref_count = cursor.fetchone()[0]
            assert ref_count > 0, "Ссылки не сохранились"
        
        print(f"Ссылки сохранены: {ref_count}")
        print("Тест ссылок пройден!")
        
    finally:
        os.unlink(db_path)


def test_idb_rst():
    """Тест 3: Сохранение .rst файлов"""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Сохранение .rst файлов")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        db = IDB(db_path)
        
        # Сохраняем .rst файл
        rst_content = """
Quickstart
==========
This is a test RST document.
"""
        doc_id = db.save_rst_document("docs/source/quickstart.rst", rst_content)
        assert doc_id > 0, "RST документ не сохранился"
        
        # Получаем необработанные
        unprocessed = db.get_unprocessed_rst_documents()
        assert len(unprocessed) == 1, "Необработанные документы не найдены"
        
        print(f"RST документ сохранен: id={doc_id}")
        print(f"Необработанных: {len(unprocessed)}")
        print("Тест RST пройден!")
        
    finally:
        os.unlink(db_path)


def test_ds1000_wrapper():
    """Тест 4: DS1000Wrapper"""
    print("\n" + "=" * 60)
    print("ТЕСТ 4: DS1000Wrapper")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        wrapper = DS1000Wrapper(db_path)
        
        # Загружаем тестовые данные
        added = wrapper.load_ds1000_sample()
        assert added > 0, "Не удалось загрузить тестовые данные"
        print(f"Загружено тестовых примеров: {added}")
        
        # Тестируем dummy_run
        results = wrapper.dummy_run("accuracy", limit=2)
        assert len(results) > 0, "dummy_run не вернул результатов"
        print(f"dummy_run нашел {len(results)} результатов")
        
        print("Тест DS1000Wrapper пройден!")
        
    finally:
        os.unlink(db_path)


def test_pagination():
    """Тест 5: Пагинация"""
    print("\n" + "=" * 60)
    print("ТЕСТ 5: Пагинация")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        db = IDB(db_path)
        
        # Добавляем несколько примеров
        examples = [create_test_example(f"test_{i}") for i in range(5)]
        db.add_extracted_examples_batch(examples, "test_repo")
        
        # Тестируем пагинацию
        page1 = db.get_examples_by_repo("test_repo", limit=2, offset=0)
        page2 = db.get_examples_by_repo("test_repo", limit=2, offset=2)
        
        assert len(page1) == 2, f"Страница 1: ожидалось 2, получено {len(page1)}"
        assert len(page2) == 2, f"Страница 2: ожидалось 2, получено {len(page2)}"
        
        print(f"Страница 1: {len(page1)} записей")
        print(f"Страница 2: {len(page2)} записей")
        print("Тест пагинации пройден!")
        
    finally:
        os.unlink(db_path)


def run_all_tests():
    """Запускает все тесты"""
    print("\n" + * 30)
    print("ЗАПУСК ВСЕХ ТЕСТОВ IDB")
    print(* 30 + "\n")
    
    tests = [
        test_idb_basic,
        test_idb_references,
        test_idb_rst,
        test_ds1000_wrapper,
        test_pagination
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print("\n Тест успешен\n" + "-" * 40)
        except AssertionError as e:
            failed += 1
            print(f"\n Ошибка в {test.__name__}: {e}\n" + "-" * 40)
        except Exception as e:
            failed += 1
            print(f"\n Исключение в {test.__name__}: {e}\n" + "-" * 40)
    
    print("\n" + "=" * 60)
    print(f"ИТОГИ: {passed} тестов пройдено,  {failed} тестов провалено")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())