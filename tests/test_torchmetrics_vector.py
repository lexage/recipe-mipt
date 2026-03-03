"""
Тест векторного поиска на реальных данных torchmetrics
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent_constructor.db import LocalDB
from src.agent_constructor.embedder import Embedder


def test_torchmetrics_vector():
    """Тестируем векторный поиск на torchmetrics"""
    

    
    # 1. Создаём эмбеддер
    
    embedder = Embedder('all-MiniLM-L6-v2')
    
    # 2. Подключаемся к БД с torchmetrics
    db_path = "data/github_examples.db"  # или путь к вашей БД
    if not os.path.exists(db_path):
        
        return
    
    
    db = LocalDB(
        embedder=embedder,
        path_to_db=db_path,
        path_to_vector_db='data/torchmetrics_vectors',
        collection_name='torchmetrics'
    )
    
    # 3. Тестовые запросы
    queries = [
        "how to calculate accuracy score",
        "precision metric for binary classification",
        "f1 score implementation",
        "recall formula",
        "auc roc metric"
    ]
    
    
    
    for query in queries:
        print(f"\nЗапрос: '{query}'")
        results = db.query(query, top_k=3)
        
        if results:
            for i, chunk in enumerate(results):
                name = chunk.metadata.get('name', 'unknown') if chunk.metadata else 'unknown'
                print(f"   {i+1}. {name}")
                print(f"      {chunk.text[:100]}...")
        else:
            print("   Ничего не найдено")
    
    # 4. Сравнение с текстовым поиском
    
    
    test_query = "accuracy metric"
    print(f"\nЗапрос: '{test_query}'")
    
    # Векторный поиск
    print("\nВЕКТОРНЫЙ ПОИСК:")
    vector_results = db.query(test_query, top_k=3)
    for i, chunk in enumerate(vector_results):
        name = chunk.metadata.get('name', 'unknown') if chunk.metadata else 'unknown'
        print(f"   {i+1}. {name}")
    
    # Текстовый поиск (через SQLite)
    print("\nТЕКСТОВЫЙ ПОИСК (по ключевым словам):")
    if hasattr(db, 'sqlite_adapter'):
        text_results = db.sqlite_adapter.search(test_query, limit=3)
        for i, chunk in enumerate(text_results):
            name = chunk.metadata.get('name', 'unknown') if chunk.metadata else 'unknown'
            print(f"   {i+1}. {name}")
    else:
        print("   Текстовый поиск недоступен")


if __name__ == "__main__":
    test_torchmetrics_vector()