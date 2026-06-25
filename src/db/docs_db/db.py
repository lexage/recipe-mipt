import re

from typing import List

from src.agent_constructor.agent import Agent
from src.agent_constructor.db import IDB
from src.agent_constructor.core import Chunk, Document, Text
from src.utils.adapters import SQLiteDocsDBAdapter, ChromaDocsAdapter, QdrantDocsAdapter
from src.utils.utils_functions import replace_examples_in_chunks


class LocalDB(IDB):
    def __init__(
            self, embedder: Agent,

            path_to_db: str = 'data/docs_database.db', 
            path_to_vector_db: str = 'data/docs_vector_database', 
            collection_name: str = 'docs',
            
            return_full_docs: bool = False,
            return_examples: bool = False,
            merge_examples: bool = False,

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
        
        self.vdb_adapter = QdrantDocsAdapter(
            embedder=embedder,
            collection_name=collection_name,
            path_to_db=path_to_vector_db,
        )

        self.return_examples = return_examples
        self.return_full_docs = return_full_docs
        self.merge_examples = merge_examples
    
    def close(self):
        """Release the underlying Qdrant storage lock (see QdrantDocsAdapter.close)."""
        adapter = getattr(self, "vdb_adapter", None)
        if adapter is not None and hasattr(adapter, "close"):
            adapter.close()

    def get_documents(self, ids: List[int] = None) -> List[Document]:
        if not ids:
            documents = self.sqlite_adapter.get_docs()
        else:
            documents = self.sqlite_adapter.get_docs(ids)
        return documents

    def query(self, query_text: Text, top_k: int = 10) -> List[Chunk]:
        chunks = self.vdb_adapter.search(query=query_text, top_k=top_k)

        if self.return_full_docs:
            doc_ids = list(set([chunk.doc_id for chunk in chunks]))
            documents : List[Document] = self.get_documents(doc_ids)
            
            chunks : List[Chunk] = []

            for doc in documents:
                metadata = doc.metadata.copy()
                metadata["source"] = doc.source

                chunks.append(
                    Chunk(
                        id=doc.id,
                        doc_id=doc.id,
                        text=doc.text,
                        metadata=metadata,
                    )
                )
        
        if self.return_examples:
            examples = []

            for chunk in chunks:
                examples_ids = self._extract_example_numbers(chunk.text)
                examples.extend(self.sqlite_adapter.get_examples_by_doc_id(chunk.doc_id, examples_ids))

            chunks.extend(examples)

        if self.merge_examples:
            chunks = replace_examples_in_chunks(chunks)

        return chunks
    
    def all_chunks(self) -> List[Chunk]:
        return self.vdb_adapter.get_chunks()
    
    def add_chunks(self, chunks: List[Chunk]):
        self.vdb_adapter.add(chunks)

    def _extract_example_numbers(self, text):
        pattern = r'<example_(\d+)>'
        matches = re.findall(pattern, text)
        numbers = [int(match) for match in matches]
        return numbers
