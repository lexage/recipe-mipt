import sqlite3
import chromadb
import json
from chromadb.config import Settings

print("🔄 Восстановление SQLite БД из векторной...")

# Подключаемся к векторной БД
client = chromadb.PersistentClient(
    path="data/github_vector_db",
    settings=Settings(anonymized_telemetry=False)
)

try:
    # Получаем коллекцию
    collection = client.get_collection("docs")
    print(f"✅ Найдена коллекция 'docs'")
    
    # Получаем все данные
    results = collection.get()
    
    print(f"📊 Всего чанков в векторной БД: {len(results['ids'])}")
    
    # Создаём SQLite БД
    conn = sqlite3.connect('data/github_example.db')
    cursor = conn.cursor()
    
    # Удаляем старую таблицу если есть и создаём новую
    cursor.execute('DROP TABLE IF EXISTS chunks')
    cursor.execute('''
    CREATE TABLE chunks (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        metadata TEXT
    )
    ''')
    
    # Восстанавливаем чанки
    restored = 0
    for i, chunk_id in enumerate(results['ids']):
        text = results['documents'][i] if results['documents'] else ""
        metadata = results['metadatas'][i] if results['metadatas'] else {}
        
        cursor.execute('''
        INSERT OR IGNORE INTO chunks (id, text, metadata)
        VALUES (?, ?, ?)
        ''', (chunk_id, text, json.dumps(metadata, ensure_ascii=False)))
        restored += 1
    
    conn.commit()
    conn.close()
    print(f"✅ Восстановлено {restored} чанков в SQLite БД")
    
    # Проверяем результат
    conn = sqlite3.connect('data/github_example.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    count = cursor.fetchone()[0]
    print(f"📊 SQLite БД теперь содержит {count} чанков")
    conn.close()
    
except Exception as e:
    print(f"❌ Ошибка: {e}")