import sqlite3
import chromadb

from typing import List
from tqdm import tqdm
from chromadb.config import Settings

from src.agent_constructor.core import Document, Text, Chunk
from src.utils.queries import GET_DOCUMENTS_QUERY, GET_EXAMPLES_QUERY


class SQLiteDocsDBAdapter:
    def __init__(self, path_to_db: str):
        self.path_to_db = path_to_db

    def get_docs(self) -> List[Document]:
        
        documents = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(GET_DOCUMENTS_QUERY)
            
            for row in cursor.fetchall():
                documents.append(
                    Document(
                        id=row['id'], 
                        source='documents', 
                        text=row['content'], metadata={
                            "library": row['library'],
                            "section": row['section'],
                            "doc_name": row['name'],
                        }
                    )
                )
        
        return documents
    
    def get_examples(self) -> List[Document]:
        documents = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(GET_EXAMPLES_QUERY)
            
            for row in cursor.fetchall():
                documents.append(
                    Document(
                        id=row['id'], 
                        source='examples', 
                        text=row['content'], 
                        metadata={
                            "doc_id": row['doc_id'],
                            "order_id": row['order_id'],
                        }
                    )
                )
        
        return documents
    

class ChromaDocsAdapter:
    def __init__(self, collection_name: str, path_to_db: str, embedding_function, bacth_size: int = 10):
        self.collection_name = collection_name
        self.path_to_db = path_to_db
        self.embedding_function = embedding_function
        self.batch_size = bacth_size

        client = chromadb.PersistentClient(path=path_to_db, settings=Settings(anonymized_telemetry=False))
        
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
            batch_metadatas = [chunk.metadata for chunk in chunks[i:i+self.batch_size]]
            self.collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metadatas)

    def search(self, queries: List[Text], top_k: int) -> List[List[Chunk]]:
        results = self.collection.query(query_texts=queries, n_results=top_k)
    
        all_query_results = []
        
        for query_idx in range(len(results['ids'])):
            query_chunks = []
            ids = results['ids'][query_idx]
            documents = results['documents'][query_idx]
            metadatas = results['metadatas'][query_idx] if results['metadatas'] else [{}] * len(ids)
            
            for i, chunk_id in enumerate(ids):
                doc_id = metadatas[i].get('doc_id', '') if i < len(metadatas) else ''
                
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
