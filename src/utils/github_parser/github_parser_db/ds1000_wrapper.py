"""
Обертка для тестирования IDB на DS-1000.
"""

import logging
from typing import List, Optional

from src.agent_constructor.core import Chunk

from .github_docs_db import GitHubDocsDB
from ..config import ExtractedExample

logger = logging.getLogger(__name__)


class DS1000Wrapper:
    """Обертка для тестирования на DS-1000."""
    
    def __init__(self, db: GitHubDocsDB):
        """
        Args:
            db: Экземпляр GitHubDocsDB для работы с данными
        """
        self.db = db
    
    def dummy_run(self, query: str = "accuracy", top_k: int = 3) -> List[Chunk]:
        """
        Тестовый запуск для проверки работоспособности.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            
        Returns:
            List[Chunk]: Найденные чанки
        """
        
        print(f"DUMMY RUN: поиск '{query}' (top_k={top_k})")
        
        
        # Выполняем поиск
        results = self.db.query(query, top_k=top_k)
        
        print(f"\nНайдено результатов: {len(results)}\n")
        
        # Выводим результаты
        for i, chunk in enumerate(results, 1):
            print(f"Результат {i}")
            print(f"ID: {chunk.id}")
            print(f"Текст: {chunk.text[:150]}" if len(chunk.text) > 150 else f"Текст: {chunk.text}")
            
            if chunk.metadata:
                print(f"Метаданные:")
                for key, value in chunk.metadata.items():
                    if value and key not in ['metadata_source_code']:
                        print(f"{key}: {str(value)[:100]}")
            print()
        
        # Статистика
        stats = self.db.get_stats()
        print("Статистика БД:")
        print(f"Всего примеров: {stats.get('total_examples', 0)}")
        print(f"Всего чанков: {stats.get('total_chunks', 0)}")
        print(f"По репозиториям: {stats.get('by_repository', {})}")
        
        return results
    
    def load_sample_data(self) -> int:
        """
        Загружает тестовые данные в стиле DS-1000.
        
        Returns:
            int: Количество загруженных примеров
        """
        logger.info("Загрузка тестовых данных DS-1000...")
        
        test_examples = [
            ExtractedExample(
                source_object_type="function",
                source_object_name="calculate_accuracy",
                source_object_path="metrics/classification.py",
                task_description="Calculate accuracy classification score",
                solution_code="def calculate_accuracy(y_true, y_pred):\n    return (y_true == y_pred).mean()",
                metadata_source_code="def calculate_accuracy(y_true, y_pred):\n    return (y_true == y_pred).mean()",
                references="https://arxiv.org/abs/1234.56789"
            ),
            ExtractedExample(
                source_object_type="function",
                source_object_name="precision_score",
                source_object_path="metrics/classification.py",
                task_description="Compute the precision",
                solution_code="def precision_score(y_true, y_pred):\n    tp = ((y_true == 1) & (y_pred == 1)).sum()\n    fp = ((y_true == 0) & (y_pred == 1)).sum()\n    return tp / (tp + fp) if (tp + fp) > 0 else 0",
                metadata_source_code="def precision_score(y_true, y_pred):\n    return tp / (tp + fp)",
                references=""
            ),
            ExtractedExample(
                source_object_type="class",
                source_object_name="Accuracy",
                source_object_path="metrics/accuracy.py",
                task_description="Accuracy metric class",
                solution_code="class Accuracy:\n    def __call__(self, y_true, y_pred):\n        return (y_true == y_pred).mean()",
                metadata_source_code="class Accuracy:\n    def __call__(self, y_true, y_pred):\n        return (y_true == y_pred).mean()",
                references=""
            ),
            ExtractedExample(
                source_object_type="function",
                source_object_name="recall_score",
                source_object_path="metrics/classification.py",
                task_description="Compute the recall",
                solution_code="def recall_score(y_true, y_pred):\n    tp = ((y_true == 1) & (y_pred == 1)).sum()\n    fn = ((y_true == 1) & (y_pred == 0)).sum()\n    return tp / (tp + fn) if (tp + fn) > 0 else 0",
                metadata_source_code="def recall_score(y_true, y_pred):\n    return tp / (tp + fn)",
                references=""
            ),
            ExtractedExample(
                source_object_type="function",
                source_object_name="f1_score",
                source_object_path="metrics/classification.py",
                task_description="Compute the F1 score",
                solution_code="def f1_score(y_true, y_pred):\n    p = precision_score(y_true, y_pred)\n    r = recall_score(y_true, y_pred)\n    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0",
                metadata_source_code="def f1_score(y_true, y_pred):\n    return 2 * p * r / (p + r)",
                references=""
            )
        ]
        
        added = self.db.add_extracted_examples(test_examples, "DS-1000")
        logger.info(f"Загружено {added} тестовых примеров")
        return added