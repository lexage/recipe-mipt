from typing import List

from src.agents.agent_constructor.chunkers import Chunker
from src.agents.agent_constructor.core import Document, Text
from src.agents.agent_constructor.db import IDB
from src.agents.agent_constructor.pipeline import Agent
from src.rag.corag.wrappers import EmbeddingFunctionWrapper
from src.utils.adapters import SQLiteDocsDBAdapter, ChromaDocsAdapter


class LocalCoragDB(IDB):
    def __init__(
            self, 
            chunker: Chunker, 
            embedding_model: Agent,
            path_to_db: str = 'data/docs_database.db', 
            path_to_vector_db: str = 'data/docs_vector_database', 
            collection_name: str = 'docs',
            ):

        self.doc_data_base = SQLiteDocsDBAdapter(
            path_to_db=path_to_db
            )

        self.vector_data_base = ChromaDocsAdapter(
            collection_name=collection_name,
            path_to_db=path_to_vector_db,
            embedding_function=EmbeddingFunctionWrapper(embedding_model)
            )
        
        documents = self.doc_data_base.get_docs()
        
        self.chunks = {
            chunk.id: chunk for chunk in self._get_chunks(
                chunker=chunker, 
                documents=documents
                )
            }

        if not self.vector_data_base.populated:
            self.vector_data_base.populate(self.all_chunks())

    def _get_chunks(self, chunker: Chunker, documents: List[Document]):
        chunks = []
        for document in documents:
            doc_chunks = chunker.chunk(document)
            chunks.extend(doc_chunks)
        return chunks
    
    def query(self, queries: Text, top_k: int):
        chunk_ids = self.vector_data_base.search(queries=[queries], top_k=top_k)          
        return [self.chunks[chunk_id] for chunk_id in chunk_ids]
    
    def all_chunks(self):
        return list(self.chunks.values())
    
    def add_chunks(self, chunks):
        return super().add_chunks(chunks)
