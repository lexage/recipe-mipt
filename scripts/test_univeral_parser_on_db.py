import sys
import sqlite3
from dataclasses import dataclass, field
from typing import (
    Any,
    Mapping,
    Optional,
    List
)
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from doc_parser.universal_parser import extract_docstrings


Text = str
Metadata = Mapping[str, Any]


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: Text
    tokens: Optional[int] = None
    metadata: Metadata = field(default_factory=dict)


class SQLiteDocsDBAdapter:
    
    def __init__(self, path_to_db: str):
        self.path_to_db = path_to_db

    def get_docs(self, library_name: str = "numpy") -> List[Chunk]:
        """Получает список чанко, принадлежащих конкретной библиотеке"""
        
        chunks = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT d.id, d.content
                FROM documents d
                JOIN sections s ON d.section_id = s.id
                JOIN libraries l ON s.library_id = l.id
                WHERE l.name = '{library_name}';
            """)
            for row in cursor.fetchall():
                chunks.append(Chunk(
                    id=row["id"],
                    doc_id=row["id"],
                    text=row["content"],
                    metadata={},
                ))
        return chunks
    
    def get_libraries_name(self):
        """Получает список имен библиотек, содержащихся в базе данных"""
        
        libraries_name = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT name FROM libraries
            """)
            for row in cursor.fetchall():
                libraries_name.append(row[0])
        return libraries_name
        

if __name__ == "__main__":
    
    path_to_db = "/workspace/data/docs_database_dedup.db"
    db = SQLiteDocsDBAdapter(path_to_db)
    
    libraries_name = db.get_libraries_name()
    print('libraries_name ', libraries_name)
    
    for library in libraries_name:
        print("*"*10)
        print(f"Парсинг библиотеки: {library}")
        print("*"*10)
        chunks = db.get_docs(library)
        strings = [chunk.text for chunk in chunks]
        docstrings = extract_docstrings(strings, library)
        if len(docstrings) > 0:
            print(f"Пример извлеченных docstrings: {docstrings[10]}")
        else:
            print(f"Не удалось получить docstrings")
            

    
        
    
    