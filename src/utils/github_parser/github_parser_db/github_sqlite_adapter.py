"""
SQLite адаптер для хранения GitHub примеров.
Конвертирует ExtractedExample в Chunk и обратно.
"""

import sqlite3
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.agent_constructor.core import Chunk
from ..config import ExtractedExample


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