"""
Тест векторного поиска с реальным эмбеддером
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent_constructor.db import LocalDB
from src.agent_constructor.core import Chunk
from src.agent_constructor.embedder import Embedder


def create_test_data():
    """Создаёт тестовые данные для поиска"""
    
    test_chunks = [
        Chunk(
            id="acc_1",
            doc_id="doc_acc",
            text="def accuracy_score(y_true, y_pred):\n    return (y_true == y_pred).mean()",
            metadata={"type": "function", "name": "accuracy_score", "library": "torchmetrics"}
        ),
        Chunk(
            id="acc_2",
            doc_id="doc_acc",
            text="class Accuracy:\n    def __call__(self, preds, target):\n        return (preds == target).float().mean()",
            metadata={"type": "class", "name": "Accuracy", "library": "torchmetrics"}
        ),
        Chunk(
            id="prec_1",
            doc_id="doc_prec",
            text="def precision_score(y_true, y_pred):\n    tp = ((y_true == 1) & (y_pred == 1)).sum()\n    fp = ((y_true == 0) & (y_pred == 1)).sum()\n    return tp / (tp + fp)",
            metadata={"type": "function", "name": "precision_score", "library": "torchmetrics"}
        ),
        Chunk(
            id="recall_1",
            doc_id="doc_recall",
            text="def recall_score(y_true, y_pred):\n    tp = ((y_true == 1) & (y_pred == 1)).sum()\n    fn = ((y_true == 1) & (y_pred == 0)).sum()\n    return tp / (tp + fn)",
            metadata={"type": "function", "name": "recall_score", "library": "torchmetrics"}
        ),
        Chunk(
            id="f1_1",
            doc_id="doc_f1",
            text="def f1_score(y_true, y_pred):\n    p = precision_score(y_true, y_pred)\n    r = recall_score(y_true, y_pred)\n    return 2 * (p * r) / (p + r)",
            metadata={"type": "function", "name": "f1_score", "library": "torchmetrics"}
        )
    ]
    
    return test_chunks


def test_vector_search_real():
    
    
    
    
    # Создаём эмбеддер

    embedder = Embedder('all-MiniLM-L6-v2')
    
    # Создаём БД
    
    db = LocalDB(
        embedder=embedder,
        path_to_db='data/test_vector.db',
        path_to_vector_db='data/test_vectors_real',
        collection_name='test_metrics'
    )
    
    
    # Шаг 3: Добавляем тестовые данные
    
    test_chunks = create_test_data()
    db.add_chunks(test_chunks)
    print(f"Добавлено {len(test_chunks)} тестовых чанков")
    
    # Шаг 4: Тестируем поиск
   
    
    test_queries = [
        "how to calculate accuracy",
        "precision metric for classification",
        "recall score implementation",
        "f1 score formula",
        "metric for binary classification"
    ]
    
    
    
    for query in test_queries:
        print(f"\nЗапрос: '{query}'")
        results = db.query(query, top_k=3)
        
        print(f"   Найдено результатов: {len(results)}")
        for i, chunk in enumerate(results):
            name = chunk.metadata.get('name', 'unknown') if chunk.metadata else 'unknown'
            print(f"   {i+1}. {name} - {chunk.text[:80]}...")


def test_on_torchmetrics_data():
    """Тестируем на реальных данных torchmetrics"""
    
    
    
    
    # Путь к вашей БД с torchmetrics
    torchmetrics_db = "data/github_examples.db"
    
    if not os.path.exists(torchmetrics_db):
        print(f"БД не найдена: {torchmetrics_db}")
        
        return
    
    # Создаём эмбеддер
    embedder = Embedder('all-MiniLM-L6-v2')
    
    # Подключаемся к БД
    db = LocalDB(
        embedder=embedder,
        path_to_db=torchmetrics_db,
        path_to_vector_db='data/torchmetrics_vectors',
        collection_name='torchmetrics'
    )
    
   
    
    # Проверяем, есть ли данные
    try:
        # Простой поиск для проверки
        test_query = "accuracy metric for classification"
        print(f"\nПоиск: '{test_query}'")
        results = db.query(test_query, top_k=5)
        print(f"Найдено результатов: {len(results)}")
        
        if len(results) > 0:
            print("\nПримеры результатов:")
            for i, chunk in enumerate(results[:3]):
                name = chunk.metadata.get('name', 'unknown') if chunk.metadata else 'unknown'
                print(f"\n   {i+1}. {name}")
                print(f"      {chunk.text[:150]}...")
    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        


if __name__ == "__main__":
    # Сначала тест на синтетических данных
    test_vector_search_real()
    
    test_on_torchmetrics_data()