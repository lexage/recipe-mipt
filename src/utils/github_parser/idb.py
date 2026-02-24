"""
ПОЛНЫЙ ИНТЕРФЕЙС БАЗЫ ДАННЫХ
Адаптирован для ветки project/github_repo_parsing
"""

import sqlite3
import hashlib
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from contextlib import contextmanager

# Импортируем модель из config Юли
from .config import ExtractedExample

logger = logging.getLogger(__name__)


class IDB:
    """
    ПОЛНЫЙ ИНТЕРФЕЙС БАЗЫ ДАННЫХ
    Поддерживает все поля из ExtractedExample
    """
    
    def __init__(self, db_path: Union[str, Path] = "code_examples.db"):
        """
        Инициализация БД.
        
        Args:
            db_path: Путь к файлу SQLite базы данных
        """
        self.db_path = Path(db_path)
        self._init_database()
        logger.info(f"IDB инициализирован: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """
        Контекстный менеджер для подключения к БД.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка при работе с БД: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Инициализирует базу данных."""
        schema_path = Path(__file__).parent / "db_schema.sql"
        
        with self.get_connection() as conn:
            if schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = f.read()
                conn.executescript(schema)
                logger.info(f"База данных инициализирована по схеме: {schema_path}")
            else:
                logger.warning("Файл схемы не найден, создаю таблицы вручную")
                self._create_tables_manually(conn)
    
    def _create_tables_manually(self, conn):
        """Создает таблицы вручную (на всякий случай)."""
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS code_examples (
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS refs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            example_id TEXT,
            reference_url TEXT NOT NULL,
            reference_title TEXT,
            reference_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (example_id) REFERENCES code_examples(id) ON DELETE CASCADE
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS rst_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            parsed_by_llm BOOLEAN DEFAULT FALSE,
            llm_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON code_examples(source_object_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_repo ON code_examples(repository_name)")
        
        conn.commit()
    
    def _generate_id(self, content: str, repository: str, path: str, name: str) -> str:
        """Генерирует уникальный ID на основе содержимого."""
        content_to_hash = f"{repository}:{path}:{name}:{content}"
        return hashlib.md5(content_to_hash.encode()).hexdigest()
    
    def add_extracted_examples_batch(self, examples: List[ExtractedExample], repo_name: str) -> int:
        """
        Сохраняет список примеров в БД.
        """
        if not examples:
            return 0
        
        added = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for example in examples:
                try:
                    example_id = self._generate_id(
                        example.solution_code,
                        repo_name,
                        example.source_object_path,
                        example.source_object_name
                    )
                    
                    cursor.execute("SELECT COUNT(*) FROM code_examples WHERE id = ?", (example_id,))
                    if cursor.fetchone()[0] > 0:
                        continue
                    
                    cursor.execute("""
                    INSERT INTO code_examples (
                        id, source_object_type, source_object_name, source_object_path,
                        task_description, solution_code, metadata_source_code, references,
                        repository_name, chunk_index, total_chunks
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        example_id,
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
                    
                    # Сохраняем ссылки отдельно
                    if example.references:
                        self._save_references(cursor, example_id, example.references)
                    
                    added += 1
                    
                except Exception as e:
                    logger.error(f"Ошибка при сохранении {example.source_object_name}: {e}")
            
            conn.commit()
        
        logger.info(f"Сохранено {added} из {len(examples)} примеров")
        return added
    
    def _save_references(self, cursor, example_id: str, references_text: str):
        """Сохраняет ссылки в отдельную таблицу."""
        import re
        urls = re.findall(r'https?://[^\s]+', references_text)
        for url in urls[:5]:
            cursor.execute("""
            INSERT INTO refs (example_id, reference_url, reference_type)
            VALUES (?, ?, ?)
            """, (example_id, url, 'unknown'))
    
    def save_rst_document(self, file_path: str, content: str) -> int:
        """Сохраняет .rst файл для LLM обработки."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO rst_docs (file_path, content, parsed_by_llm)
            VALUES (?, ?, FALSE)
            """, (file_path, content))
            conn.commit()
            return cursor.lastrowid
    
    def get_unprocessed_rst_documents(self) -> List[Dict[str, Any]]:
        """Получает .rst файлы для обработки LLM."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rst_docs WHERE parsed_by_llm = FALSE")
            return [dict(row) for row in cursor.fetchall()]
    
    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Поиск по примерам."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM code_examples 
            WHERE task_description LIKE ? 
               OR solution_code LIKE ?
               OR source_object_name LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """, (f'%{query}%', f'%{query}%', f'%{query}%', limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика по БД."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM code_examples")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM rst_docs")
            rst_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM refs")
            ref_count = cursor.fetchone()[0]
            
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                'total_examples': total,
                'rst_documents': rst_count,
                'references': ref_count,
                'database_size_mb': db_size / (1024 * 1024)
            }


class DS1000Wrapper:
    """Обертка для DS-1000."""
    
    def __init__(self, db_path: str = "ds1000.db"):
        self.db = IDB(db_path)
        self.logger = logging.getLogger(__name__)
    
    def dummy_run(self, query: str = "accuracy", limit: int = 3) -> List[Dict[str, Any]]:
        """Тестовый запуск."""
        self.logger.info(f"DUMMY RUN: поиск '{query}'")
        results = self.db.search(query, limit=limit)
        self.logger.info(f"Найдено {len(results)} результатов")
        return results