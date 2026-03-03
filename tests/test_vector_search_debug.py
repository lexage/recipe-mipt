"""
Отладочная версия теста векторного поиска
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent_constructor.db import LocalDB
from src.agent_constructor.core import Chunk
from src.agent_constructor.embedder import Embedder


def debug_vector_search():
    """Отладочная версия с проверкой каждого шага"""
    
    
    #Создаём эмбеддер
    
    embedder = Embedder('all-MiniLM-L6-v2')
    
    # Создаём тестовые данные
    
    test_chunks = [
        Chunk(
            id="acc_1",
            doc_id="doc_acc",
            text="def accuracy_score(y_true, y_pred): return (y_true == y_pred).mean()",
            metadata={"name": "accuracy_score", "type": "function"}
        ),
        Chunk(
            id="acc_2",
            doc_id="doc_acc",
            text="class Accuracy: def __call__(self, preds, target): return (preds == target).float().mean()",
            metadata={"name": "Accuracy", "type": "class"}
        )
    ]
    print(f"   Создано {len(test_chunks)} чанков")
    for chunk in test_chunks:
        print(f"   - {chunk.id}: {chunk.metadata['name']}")
    
    # Создаём БД
    
    db = LocalDB(
        embedder=embedder,
        path_to_db='data/test_debug.db',
        path_to_vector_db='data/test_vectors_debug',
        collection_name='debug'
    )
    
    # Добавляем данные
    
    db.add_chunks(test_chunks)
    
    # Проверяем, что данные добавились
    
    try:
        # Пробуем получить все чанки (если есть такой метод)
        if hasattr(db, 'all_chunks'):
            all_chunks = db.all_chunks()
            print(f"   all_chunks() вернул {len(all_chunks)} чанков")
        
        # Проверяем напрямую через адаптер
        if hasattr(db, 'vdb_adapter'):
            chunks_in_vdb = db.vdb_adapter.get_chunks()
            print(f"   В векторной БД: {len(chunks_in_vdb)} чанков")
    except Exception as e:
        print(f"   Ошибка при проверке: {e}")
    
    # Пробуем поиск
    
    test_queries = [
        "accuracy",
        "precision",
        "classification"
    ]
    
    for query in test_queries:
        print(f"\nЗапрос: '{query}'")
        try:
            results = db.query(query, top_k=3)
            print(f"   Результатов: {len(results)}")
            
            if results:
                for i, chunk in enumerate(results):
                    name = chunk.metadata.get('name', 'unknown') if chunk.metadata else 'unknown'
                    print(f"   {i+1}. {name}: {chunk.text[:50]}...")
            else:
                print(" Нет результатов")
                
                
                if hasattr(db, 'vdb_adapter'):
                    print(" Пробуем прямой поиск через Chroma...")
                    try:
                        # Получаем эмбеддинг запроса
                        query_embedding = embedder(query)
                        
                        print("      Не реализовано в этой версии")
                    except Exception as e2:
                        print(f"      Ошибка: {e2}")
                        
        except Exception as e:
            print(f" Ошибка при поиске: {e}")
    
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path='data/test_vectors_debug')
        collections = client.list_collections()
        print(f"   Коллекций в Chroma: {len(collections)}")
        for coll in collections:
            print(f"   - {coll.name}: {coll.count()} элементов")
    except Exception as e:
        print(f"   Ошибка при проверке Chroma: {e}")


if __name__ == "__main__":
    debug_vector_search()