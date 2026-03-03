"""
Адаптеры для работы с базами данных.

"""

import sqlite3
import json
import hashlib
import time
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.utils.github_parser.config import ExtractedExample

class GitHubSQLiteAdapter:
    """
    Адаптер для работы с существующей SQLite БД.
    
    """
    
    def __init__(self, db_path: str = "data/docs_database.db"):
        """
        Инициализация адаптера с существующей БД.
        
        Args:
            db_path: Путь к существующей SQLite базе данных
        """
        self.db_path = db_path
        self._connections = []
        
        # Проверяем, что файл существует
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"База данных не найдена: {db_path}")
        
        print(f"Подключено к существующей БД: {db_path}")
    
    def _get_connection(self):
        """Создаёт соединение с БД."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        self._connections.append(conn)
        return conn
    
    def close_all_connections(self):
        """Закрывает все соединения."""
        for conn in self._connections:
            try:
                conn.close()
            except:
                pass
        self._connections.clear()
    
    def table_exists(self, table_name: str) -> bool:
        """
        Проверяет, существует ли таблица в БД.
        
        Args:
            table_name: Имя таблицы
            
        Returns:
            bool: True если таблица существует
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        exists = cursor.fetchone() is not None
        return exists
    
    def add_example(self, example, repo_name: str) -> str:
        """
        Добавляет пример в существующую БД.
        Если таблиц нет - использует то, что есть.
        """
        chunk_id = self._generate_chunk_id(example, repo_name)
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Проверяем, есть ли таблица chunks
        if self.table_exists('chunks'):
            cursor.execute("""
            INSERT OR IGNORE INTO chunks (id, text, metadata)
            VALUES (?, ?, ?)
            """, (
                chunk_id,
                example.solution_code,
                json.dumps({
                    'source_object_type': example.source_object_type,
                    'source_object_name': example.source_object_name,
                    'source_object_path': example.source_object_path,
                    'repository_name': repo_name
                }, ensure_ascii=False)
            ))
        
        conn.commit()
        return chunk_id
    
    def _generate_chunk_id(self, example, repo_name: str) -> str:
        """Генерирует ID."""
        content = f"{repo_name}:{example.source_object_path}:{example.source_object_name}:{example.solution_code}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_examples_batch(self, examples, repo_name: str) -> int:
        """Добавляет несколько примеров."""
        added = 0
        for example in examples:
            try:
                self.add_example(example, repo_name)
                added += 1
            except Exception as e:
                print(f"Ошибка: {e}")
        return added
    
    def search(self, query: str, limit: int = 10) -> List:
        """Поиск по существующей БД."""
        from src.agent_constructor.core import Chunk
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Пробуем поиск в chunks если есть
        if self.table_exists('chunks'):
            cursor.execute("""
            SELECT id, text, metadata FROM chunks
            WHERE text LIKE ? OR metadata LIKE ?
            LIMIT ?
            """, (f'%{query}%', f'%{query}%', limit))
            
            chunks = []
            for row in cursor.fetchall():
                chunk = Chunk(
                    id=row[0],
                    text=row[1],
                    metadata=json.loads(row[2]) if row[2] else None
                )
                chunks.append(chunk)
            return chunks
        
        return []
    
    def get_stats(self) -> Dict:
        """Статистика по существующей БД."""
        stats = {
            'database_path': str(self.db_path),
            'tables': []
        }
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Получаем список таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        stats['tables'] = [t[0] for t in tables]
        
        # Считаем записи в каждой таблице
        for table in stats['tables']:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f'{table}_count'] = count
            except:
                pass
        
        return stats
    

class SQLiteDocsDBAdapter:
    """
    Адаптер для работы с SQLite документами.
    Соответствует тому, что ожидает agent_constructor.
    """
    
    def __init__(self, path_to_db: str = "data/docs_database.db"):
        """
        Инициализация адаптера.
        
        Args:
            path_to_db: Путь к SQLite базе данных
        """
        self.path_to_db = path_to_db
        self._init_database()
        print(f"SQLiteDocsDBAdapter подключен к {path_to_db}")
    
    def _init_database(self):
        """Инициализация таблиц, если их нет."""
        conn = sqlite3.connect(self.path_to_db)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            code TEXT,
            description TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        conn.close()
    
    def get_docs(self, ids: Optional[List[int]] = None) -> List:
        """Получает документы."""
        from src.agent_constructor.core import Document
        
        conn = sqlite3.connect(self.path_to_db)
        cursor = conn.cursor()
        
        if ids:
            placeholders = ','.join(['?'] * len(ids))
            cursor.execute(f"SELECT * FROM documents WHERE id IN ({placeholders})", ids)
        else:
            cursor.execute("SELECT * FROM documents")
        
        docs = []
        for row in cursor.fetchall():
            # Создаем Document объект
            doc = Document(
                id=str(row[0]),
                title=row[1] or "",
                content=row[2] or "",
                metadata=json.loads(row[3]) if row[3] else {}
            )
            docs.append(doc)
        
        conn.close()
        return docs
    
    def get_examples(self) -> List:
        """Получает примеры."""
        conn = sqlite3.connect(self.path_to_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM examples")
        
        examples = []
        for row in cursor.fetchall():
            examples.append({
                'id': row[0],
                'name': row[1],
                'code': row[2],
                'description': row[3],
                'metadata': json.loads(row[4]) if row[4] else {}
            })
        
        conn.close()
        return examples


class ChromaDocsAdapter:
    """
    Адаптер для работы с векторной БД Chroma.
    Реализация с ChromaDB.
    """
    
    def __init__(self, embedder=None, collection_name: str = "docs", path_to_db: str = "data/docs_vector_database"):
        """
        Инициализация адаптера.
        
        Args:
            embedder: Модель для создания эмбеддингов
            collection_name: Название коллекции
            path_to_db: Путь к векторной БД
        """
        self.embedder = embedder
        self.collection_name = collection_name
        self.path_to_db = path_to_db
        self.client = None
        self.collection = None
        self._chunks = []
        
        self._init_chroma()
        print(f"ChromaDocsAdapter инициализирован: {path_to_db}")
    
    def _init_chroma(self):
        """Инициализация подключения к ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Создаём клиент
            self.client = chromadb.PersistentClient(path=self.path_to_db)
            
            # Получаем или создаём коллекцию
            try:
                self.collection = self.client.get_collection(self.collection_name)
                print(f"   Коллекция '{self.collection_name}' найдена, элементов: {self.collection.count()}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"   Создана новая коллекция '{self.collection_name}'")
                
        except ImportError:
            print("chromadb не установлен. Установите: pip install chromadb")
        except Exception as e:
            print(f"Ошибка при инициализации Chroma: {e}")
    
    def add(self, chunks):
        """
        Добавляет чанки в векторную БД.
        
        Args:
            chunks: Список чанков для добавления
        """
        if not chunks:
            return
        
        if not self.collection:
            print("Коллекция Chroma не инициализирована")
            self._chunks.extend(chunks)
            return
        
        try:
            # Подготавливаем данные для Chroma
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for chunk in chunks:
                ids.append(chunk.id)
                
                # Получаем эмбеддинг через embedder
                if self.embedder:
                    embedding = self.embedder(chunk.text)
                    embeddings.append(embedding)
                else:
                    print(f"Нет embedder'а, пропускаем чанк {chunk.id}")
                    continue
                
                metadatas.append(chunk.metadata or {})
                documents.append(chunk.text)
            
            if not embeddings:
                print("Нет эмбеддингов для добавления")
                return
            
            # Добавляем в коллекцию
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            # Сохраняем в локальном кэше
            self._chunks.extend(chunks)
            print(f"Добавлено {len(chunks)} чанков в ChromaDocsAdapter")
            
        except Exception as e:
            print(f"Ошибка при добавлении в Chroma: {e}")
            import traceback
            traceback.print_exc()
            # На всякий случай сохраняем в локальном кэше
            self._chunks.extend(chunks)
    
    def search(self, queries=None, top_k: int = 10) -> List[List]:
        """
        Поиск по векторной БД.
        
        Args:
            queries: Список запросов
            top_k: Количество результатов
            
        Returns:
            List[List]: Список результатов для каждого запроса
        """
        from src.agent_constructor.core import Chunk
        
        if not queries:
            return [[]]
        
        if not self.collection:
            print("Коллекция Chroma не инициализирована, возвращаем пустые результаты")
            return [[] for _ in queries]
        
        try:
            results_list = []
            
            for query in queries:
                # Получаем эмбеддинг запроса
                if not self.embedder:
                    print("Нет embedder'а для поиска")
                    results_list.append([])
                    continue
                
                query_embedding = self.embedder(query)
                
                # Ищем в коллекции
                try:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(top_k, self.collection.count())
                    )
                    
                    chunks = []
                    if results['ids'] and len(results['ids'][0]) > 0:
                        for i, doc_id in enumerate(results['ids'][0]):
                            chunk = Chunk(
                                id=doc_id,
                                doc_id=doc_id,
                                text=results['documents'][0][i] if results['documents'] else "",
                                metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                            )
                            # Добавляем score если есть (конвертируем расстояние в сходство)
                            if results.get('distances'):
                                # distance = 0 (близко) -> score = 1
                                # distance = 2 (далеко) -> score = 0
                                chunk.score = 1 - (results['distances'][0][i] / 2)
                            chunks.append(chunk)
                    
                    results_list.append(chunks)
                    
                except Exception as e:
                    print(f"Ошибка при поиске запроса '{query}': {e}")
                    results_list.append([])
            
            return results_list
            
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            return [[] for _ in queries]
    
    def get_chunks(self) -> List:
        """Получает все чанки из локального кэша."""
        return self._chunks
    
    def get_all_from_chroma(self) -> List:
        """Получает все чанки напрямую из Chroma."""
        from src.agent_constructor.core import Chunk
        
        if not self.collection:
            return []
        
        try:
            results = self.collection.get()
            
            chunks = []
            for i, doc_id in enumerate(results['ids']):
                chunk = Chunk(
                    id=doc_id,
                    doc_id=doc_id,
                    text=results['documents'][i] if results['documents'] else "",
                    metadata=results['metadatas'][i] if results['metadatas'] else {}
                )
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            print(f"Ошибка при получении данных из Chroma: {e}")
            return []