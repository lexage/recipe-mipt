import sqlite3

from typing import List
from tqdm import tqdm

from src.rag.raptor.core import RetrievalAugmentation
from src.agents.agent_constructor.chunkers import Chunker
from src.agents.agent_constructor.core import Document, Chunk, Text
from src.agents.agent_constructor.db import IDB
from src.rag.corag.models import TFIDFEmbeddingFunction


BATCH_SIZE = 10


class SQLiteDataBase:
    def __init__(self, path_to_db: str):
        self.path_to_db = path_to_db

    def get_docs(self) -> List[Document]:
        
        documents = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT d.id, d.filename, d.content, d.file_path, 
                    s.name as section, l.name as library
                FROM documents d
                JOIN sections s ON d.section_id = s.id
                JOIN libraries l ON s.library_id = l.id
                WHERE section == "user_guide"
            ''')
            
            for row in cursor.fetchall():
                documents.append(Document(id=row['id'], source=row['library'], text=row['content'], metadata={}))
        
        return documents


class LocalRAPTORDB(IDB):
    def __init__(
            self, 
            chunker: Chunker, 
            path_to_db: str = 'data/docs_database.db', 
            ):

        self.doc_data_base = SQLiteDataBase(
            path_to_db=path_to_db
            )
        
        documents = self.doc_data_base.get_docs()
        
        self.chunks = {
            chunk.id: chunk for chunk in self._get_chunks(
                chunker=chunker, 
                documents=documents
                )
            }

    def _get_chunks(self, chunker: Chunker, documents: List[Document]):
        chunks = []
        for document in documents:
            doc_chunks = chunker.chunk(document)
            chunks.extend(doc_chunks)
        return chunks
    
    def query(self, queries: Text, top_k: int):
        pass

    def all_chunks(self):
        return list(self.chunks.values())
    
    def add_chunks(self, chunks):
        return super().add_chunks(chunks)
