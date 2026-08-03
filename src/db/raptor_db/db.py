from typing import List

from src.agent_constructor.chunkers import Chunker
from src.agent_constructor.db import IDB
from src.agent_constructor.core import Document, Text

from src.utils.adapters import SQLiteDocsDBAdapter


class LocalRaptorDB(IDB):
    def __init__(
            self, 
            chunker: Chunker, 
            path_to_db: str = 'data/docs_database.db', 
            ):

        self.doc_data_base = SQLiteDocsDBAdapter(
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
        pass
