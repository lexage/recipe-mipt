import sqlite3
import chromadb
import json
import hashlib

from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
from chromadb.config import Settings

from src.utils import DOCUMENT_SRC_DOCUMENTS, DOCUMENT_SRC_EXAMPLES
from src.agent_constructor.core import Document, Text, Chunk
from src.utils.queries import GET_DOCUMENTS_QUERY, GET_EXAMPLES_QUERY
from src.utils.wrappers import EmbeddingFunctionWrapper

from ..config import ExtractedExample


class SQLiteDocsDBAdapter:
    def __init__(self, path_to_db: str):
        self.path_to_db = path_to_db

    def get_docs(self, ids: Optional[List[int]] = None) -> List[Document]:
        
        documents = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if ids:
                placeholders = ','.join('?' * len(ids))
                query = f"{GET_DOCUMENTS_QUERY} WHERE d.id IN ({placeholders})"
                cursor.execute(query, ids)
            else:
                cursor.execute(GET_DOCUMENTS_QUERY)
            
            for row in cursor.fetchall():
                documents.append(
                    Document(
                        id=row['id'], 
                        source=DOCUMENT_SRC_DOCUMENTS, 
                        text=row['content'], metadata={
                            "library": row['library'],
                            "section": row['section'],
                            "doc_name": row['name'],
                        }
                    )
                )
        
        return documents
    
    def get_examples(self, ids: Optional[List[int]] = None) -> List[Document]:
        documents = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            id_offset = cursor.fetchone()[0]
            
            if ids:
                placeholders = ','.join('?' * len(ids))
                query = f"{GET_EXAMPLES_QUERY} WHERE e.id IN ({placeholders})"
                cursor.execute(query, ids)
            else:
                cursor.execute(GET_EXAMPLES_QUERY)
            
            for row in cursor.fetchall():
                documents.append(
                    Document(
                        id=row['id'] + id_offset, 
                        source=DOCUMENT_SRC_EXAMPLES, 
                        text=row['content'], 
                        metadata={
                            "doc_id": row['doc_id'],
                            "order_id": row['order_id'],
                        }
                    )
                )
        
        return documents
    

class ChromaDocsAdapter:
    
    def __init__(self, embedder, collection_name: str, path_to_db: str, batch_size: int = 10):
        self.collection_name = collection_name
        self.path_to_db = path_to_db
        self.batch_size = batch_size

        client = chromadb.PersistentClient(path=path_to_db, settings=Settings(anonymized_telemetry=False))
        
        self.collection = client.get_or_create_collection(
            self.collection_name, 
            embedding_function=EmbeddingFunctionWrapper(embedder)
        )

    def _get_existing_ids(self) -> set:
        existing = self.collection.get(include=[])
        return set(existing['ids']) if existing['ids'] else set()

    def _filter_new_chunks(self, chunks: List[Chunk], existing_ids: set) -> List[Chunk]:
        return [chunk for chunk in chunks if chunk.id not in existing_ids]

    def add(self, chunks: List[Chunk]):
        existing_ids = self._get_existing_ids()
        new_chunks = self._filter_new_chunks(chunks, existing_ids)
        
        if not new_chunks:
            return
        
        for i in tqdm(range(0, len(new_chunks), self.batch_size), desc="Vectorizing"):
            batch = new_chunks[i:i + self.batch_size]
            
            batch_docs = [chunk.text for chunk in batch]
            batch_ids = [chunk.id for chunk in batch]
            batch_metadatas = [chunk.metadata for chunk in batch]
            
            self.collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metadatas
            )

    def get_chunks(self, ids: Optional[List[str]] = None) -> List[Chunk]:
        if ids:
            results = self.collection.get(ids=ids, include=['documents', 'metadatas'])
        else:
            results = self.collection.get(include=['documents', 'metadatas'])
        
        chunks = []
        result_ids = results['ids'] if results['ids'] else []
        documents = results['documents'] if results['documents'] else []
        metadatas = results['metadatas'] if results['metadatas'] else []
        
        for i, chunk_id in enumerate(result_ids):
            doc_id = metadatas[i].get('doc_id', None) if i < len(metadatas) else None
            
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=documents[i] if i < len(documents) else '',
                    tokens=None,
                    metadata=metadatas[i] if i < len(metadatas) else {}
                )
            )
        
        return chunks

    def search(self, queries: List[Text], top_k: int) -> List[List[Chunk]]:

        results = self.collection.query(query_texts=queries, n_results=top_k)
    
        all_query_results = []
        
        for query_idx in range(len(results['ids'])):
            query_chunks = []
            ids = results['ids'][query_idx]
            documents = results['documents'][query_idx]
            metadatas = results['metadatas'][query_idx] if results['metadatas'] else [{}] * len(ids)
            
            for i, chunk_id in enumerate(ids):
                doc_id = metadatas[i].get('doc_id', None) if i < len(metadatas) else None
                
                query_chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        text=documents[i] if i < len(documents) else '',
                        tokens=None,
                        metadata=metadatas[i] if i < len(metadatas) else {}
                    )
                )
            
            all_query_results.append(query_chunks)
        
        return all_query_results

"""
SQLite адаптер для хранения GitHub примеров.
Конвертирует ExtractedExample в Chunk и обратно.
"""


class GitHubSQLiteAdapter:
    """Адаптер для работы с SQLite БД GitHub примеров."""
    
    def __init__(self, db_path: str = "data/github_examples.db"):
        self.db_path = db_path
        # Создаем папку data, если её нет
        Path("data").mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Инициализирует БД с нужными таблицами."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблица для chunks
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Таблица для сырых ExtractedExample (на всякий случай)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_examples (
                id TEXT PRIMARY KEY,
                source_object_type TEXT NOT NULL,
                source_object_name TEXT NOT NULL,
                source_object_path TEXT NOT NULL,
                task_description TEXT,
                solution_code TEXT NOT NULL,
                metadata_source_code TEXT,
                references TEXT,
                repository_name TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Индексы для поиска
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_text ON chunks(text)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_type ON raw_examples(source_object_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_repo ON raw_examples(repository_name)")
            
            conn.commit()
    
    def _generate_chunk_id(self, example: ExtractedExample, repo_name: str) -> str:
        """Генерирует ID для чанка."""
        content = f"{repo_name}:{example.source_object_path}:{example.source_object_name}:{example.solution_code}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def example_to_chunk(self, example: ExtractedExample, repo_name: str) -> Chunk:
        """Конвертирует ExtractedExample в Chunk."""
        metadata = {
            'source_object_type': example.source_object_type,
            'source_object_name': example.source_object_name,
            'source_object_path': example.source_object_path,
            'task_description': example.task_description,
            'repository_name': repo_name,
            'references': example.references,
            'metadata_source_code': example.metadata_source_code[:200] if example.metadata_source_code else ""  # обрезаем для метаданных
        }
        
        return Chunk(
            id=self._generate_chunk_id(example, repo_name),
            text=example.solution_code,
            metadata=metadata
        )
    
    def add_example(self, example: ExtractedExample, repo_name: str) -> str:
        """Добавляет пример в БД (и в chunks, и в raw_examples)."""
        chunk_id = self._generate_chunk_id(example, repo_name)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Сохраняем в chunks
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
                    'task_description': example.task_description,
                    'repository_name': repo_name,
                    'references': example.references
                }, ensure_ascii=False)
            ))
            
            # Сохраняем в raw_examples
            cursor.execute("""
            INSERT OR IGNORE INTO raw_examples (
                id, source_object_type, source_object_name, source_object_path,
                task_description, solution_code, metadata_source_code, references,
                repository_name, chunk_index, total_chunks
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id,
                example.source_object_type,
                example.source_object_name,
                example.source_object_path,
                example.task_description or "",
                example.solution_code,
                example.metadata_source_code or "",
                example.references or "",
                repo_name,
                0, 1
            ))
            
            conn.commit()
            return chunk_id
    
    def add_examples_batch(self, examples: List[ExtractedExample], repo_name: str) -> int:
        """Добавляет несколько примеров."""
        added = 0
        for example in examples:
            try:
                self.add_example(example, repo_name)
                added += 1
            except Exception as e:
                print(f"Ошибка при добавлении {example.source_object_name}: {e}")
        return added
    
    def search(self, query: str, limit: int = 10) -> List[Chunk]:
        """Поиск по тексту чанков."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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
    
    def get_all_chunks(self) -> List[Chunk]:
        """Возвращает все чанки."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, text, metadata FROM chunks")
            
            chunks = []
            for row in cursor.fetchall():
                chunk = Chunk(
                    id=row[0],
                    text=row[1],
                    metadata=json.loads(row[2]) if row[2] else None
                )
                chunks.append(chunk)
            return chunks
    
    def get_examples_by_repo(self, repo_name: str, limit: int = 100) -> List[Dict]:
        """Получает сырые примеры по репозиторию."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM raw_examples 
            WHERE repository_name = ?
            ORDER BY created_at DESC
            LIMIT ?
            """, (repo_name, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика по БД."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM raw_examples")
            total_examples = cursor.fetchone()[0]
            
            cursor.execute("""
            SELECT repository_name, COUNT(*) 
            FROM raw_examples 
            GROUP BY repository_name
            """)
            repo_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute("""
            SELECT source_object_type, COUNT(*) 
            FROM raw_examples 
            GROUP BY source_object_type
            """)
            type_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_chunks': total_chunks,
                'total_examples': total_examples,
                'by_repository': repo_stats,
                'by_type': type_stats,
                'database_path': str(self.db_path)
            }