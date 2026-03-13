from typing import List
import sys
import os

# Добавляем корень проекта в путь, чтобы импорты работали
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



try:
    from src.agent_constructor.db import IDB
    from src.agent_constructor.core import Document, Chunk
    
except Exception as e:
    print(f"Ошибка импорта agent_constructor: {e}")
    sys.exit(1)

try:
    from src.agents.general.embedding_agents import EmbeddingAgent
    
except Exception as e:
    print(f"Ошибка импорта EmbeddingAgent: {e}")
    sys.exit(1)

try:
    from src.agent_constructor.context_engine import SimpleContextAssembler
    from src.agent_constructor.chunkers import DummyChunker
    
except Exception as e:
    print(f"Ошибка импорта context_engine/chunkers: {e}")
    sys.exit(1)

try:
    from src.utils.github_parser.github_parser_db.github_docs_db import GitHubDocsDB
    
except Exception as e:
    print(f"Ошибка импорта GitHubDocsDB: {e}")
    sys.exit(1)


try:
    embedder = EmbeddingAgent(
        url="http://localhost:7216/v1",
        model_name="Qwen/Qwen3-Embedding-4B",
    )
    print(f"EmbeddingAgent создан: {embedder}")
except Exception as e:
    print(f"Ошибка создания EmbeddingAgent: {e}")
    sys.exit(1)


db_path = "data/github_example.db"
vector_db_path = "data/github_vector_db"

# Проверяем, существует ли файл БД
if not os.path.exists(db_path):
    print(f" Ошибка: БД не найдена по пути {db_path}")
    
    sys.exit(1)
else:
    file_size = os.path.getsize(db_path)
    print(f"Файл БД найден: {db_path} (размер: {file_size} байт)")
    if file_size < 100000:
        print(" БД маленькая. Возможно, данные ещё не сохранены.")

try:
    db: IDB = GitHubDocsDB(
        embedder=embedder,
        db_path=db_path,
        vector_db_path=vector_db_path,
        collection_name="docs",
    )
    
except Exception as e:
    print(f"Ошибка создания GitHubDocsDB: {e}")
    sys.exit(1)


try:
    # Проверяем, есть ли чанки в векторной БД
    all_chunks = db.all_chunks()
    print(f" Всего чанков в векторной БД: {len(all_chunks)}")
    
    if len(all_chunks) > 0:
        print(f"\n Первый чанк:")
        print(f"   ID: {all_chunks[0].id}")
        print(f"   Текст: {all_chunks[0].text[:150]}...")
        if all_chunks[0].metadata:
            print(f"   Метаданные: {all_chunks[0].metadata}")
    else:
        print("Векторная БД пуста. Возможно, нужно запустить парсер для сохранения данных.")
        
except Exception as e:
    print(f"Ошибка при проверке чанков: {e}")


try:
    chuker = DummyChunker()
    assembler = SimpleContextAssembler(name="sad")
    
except Exception as e:
    print(f"Ошибка создания компонентов: {e}")
    sys.exit(1)




query = "How to merge numpy array?"
print(f"Запрос: '{query}'")
try:
    result: List[Chunk] = db.query(query_text=query, top_k=5)
    print(f"Поиск выполнен, найдено результатов: {len(result)}")
    
    if result:
        print("\nРЕЗУЛЬТАТЫ ПОИСКА:")
        for i, chunk in enumerate(result):
            print(f"\n--- Результат {i+1} ---")
            name = chunk.metadata.get('name', 'unknown') if chunk.metadata else 'unknown'
            obj_type = chunk.metadata.get('type', 'unknown') if chunk.metadata else 'unknown'
            print(f"   Имя: {name}")
            print(f"   Тип: {obj_type}")
            print(f"   Текст: {chunk.text[:200]}..." if len(chunk.text) > 200 else f"   Текст: {chunk.text}")
            if chunk.metadata:
                print(f"   Метаданные: {chunk.metadata}")
    else:
        print("Результатов не найдено. Попробуйте другой запрос.")
        
except Exception as e:
    print(f"Ошибка при поиске: {e}")
    sys.exit(1)


try:
    context = assembler.assemble(result)
    
except Exception as e:
    print(f"Ошибка при сборке контекста: {e}")
    sys.exit(1)


print(context)
