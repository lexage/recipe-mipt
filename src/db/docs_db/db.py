from typing import List

from src.agent_constructor.agent import Agent
from src.agent_constructor.db import IDB
from src.agent_constructor.core import Chunk, Document, Text
from src.utils.adapters import SQLiteDocsDBAdapter, ChromaDocsAdapter


class LocalDB(IDB):
    def __init__(
            self, embedder: Agent,

            path_to_db: str = 'data/docs_database.db', 
            path_to_vector_db: str = 'data/docs_vector_database', 
            collection_name: str = 'docs',
            
            hnsw_space: str = "cosine",
            hnsw_m: int = 16,
            hnsw_construction_ef: int = 400,
            hnsw_search_ef: int = 200,
            batch_size: int = 16,

            similarity_threshold: float = 0.1,
            
            search_filter: dict = {}):

        self.sqlite_adapter = SQLiteDocsDBAdapter(
            path_to_db=path_to_db
            )
        
        self.vdb_adapter = ChromaDocsAdapter(
            embedder=embedder,
            collection_name=collection_name,
            path_to_db=path_to_vector_db,
            hnsw_space=hnsw_space,
            hnsw_m=hnsw_m,
            hnsw_construction_ef=hnsw_construction_ef,
            hnsw_search_ef=hnsw_search_ef,
            batch_size=batch_size,
            search_filter=search_filter,
            threshold=similarity_threshold,
        )
    
    def get_documents(self, ids: List[int] = None) -> List[Document]:
        if not ids:
            documents = self.sqlite_adapter.get_docs()
            documents.extend(self.sqlite_adapter.get_examples())
        else:
            documents = self.sqlite_adapter.get_docs(ids)
        return documents

    def query(self, query_text: Text, top_k: int = 10) -> List[Chunk]:
        chunks = self.vdb_adapter.search(queries=[query_text], top_k=top_k)[0]      
        return chunks
    
    def all_chunks(self) -> List[Chunk]:
        return self.vdb_adapter.get_chunks()
    
    def add_chunks(self, chunks: List[Chunk]):
        self.vdb_adapter.add(chunks)
