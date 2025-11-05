import sqlite3
import chromadb

from typing import List
from tqdm import tqdm

from src.agent_constructor.core import Document, Text, Chunk


class SQLiteDocsDBAdapter:
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
            ''')
            
            for row in cursor.fetchall():
                documents.append(Document(id=row['id'], source=row['library'], text=row['content'], metadata={}))
        
        return documents
    

class ChromaDocsAdapter:
    def __init__(self, collection_name: str, path_to_db: str, embedding_function, bacth_size: int = 10):
        self.collection_name = collection_name
        self.path_to_db = path_to_db
        self.embedding_function = embedding_function
        self.batch_size = bacth_size

        client = chromadb.PersistentClient(path=path_to_db)
        
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        self.populated = collection_name in collection_names
        
        if self.populated:
            self.collection = client.get_or_create_collection(
                self.collection_name, 
                embedding_function=self.embedding_function
            )
        self.client = client

    def populate(self, chunks: List[Chunk]):
        self.collection = self.client.get_or_create_collection(
            self.collection_name, 
            embedding_function=self.embedding_function
        )

        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Vectorising"):

            batch_docs = [chunk.text for chunk in chunks[i:i+self.batch_size]]
            batch_ids = [chunk.id for chunk in chunks[i:i+self.batch_size]]
            self.collection.add(documents=batch_docs, ids=batch_ids)

    def search(self, queries: List[Text], top_k: int):
        results = self.collection.query(query_texts=queries, n_results=top_k)
        return results.get("ids", [[]])[0]
