import sqlite3
import chromadb

from typing import List, Optional
from tqdm import tqdm
from chromadb.config import Settings

from src.agent_constructor.core import Document, Text, Chunk
from src.utils.queries import GET_DOCUMENTS_QUERY, GET_EXAMPLES_QUERY
from src.utils.wrappers import EmbeddingFunctionWrapper


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
                        source='documents', 
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
            
            if ids:
                placeholders = ','.join('?' * len(ids))
                query = f"{GET_EXAMPLES_QUERY} WHERE e.id IN ({placeholders})"
                cursor.execute(query, ids)
            else:
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
