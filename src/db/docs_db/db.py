import re

from typing import List

from src.agent_constructor.agent import Agent
from src.agent_constructor.db import IDB
from src.agent_constructor.core import Chunk, Document, Text
from src.db.json_corpus import resolve_sqlite_path
from src.utils import DOCUMENT_SRC_DOCUMENTS, DOCUMENT_SRC_EXAMPLES
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

            max_docs: int = None,
            max_docs_tail: int = 0,

            index_sources: List[str] = None,

            search_filter: dict = {}):

        self.max_docs = max_docs
        self.max_docs_tail = max_docs_tail

        # Which SQLite tables get loaded and embedded into the vector index at
        # build time. Default = documents-only (backward compatible with exp<=6).
        # Values: any subset of ['documents', 'examples'].
        self.index_sources = index_sources or [DOCUMENT_SRC_DOCUMENTS]

        # A JSON corpus (.json / .json.gz) is validated and converted to SQLite
        # once; everything below keeps reading SQLite as before.
        path_to_db = resolve_sqlite_path(path_to_db)

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
        self.vdb_adapter.close()

    def get_documents(self, ids: List[int] = None) -> List[Document]:
        # ids-path (used by return_full_docs at query time) keeps the historical
        # documents-only behaviour; index_sources only governs the build-time
        # corpus load (ids is None).
        if ids:
            return self.sqlite_adapter.get_docs(ids)

        documents: List[Document] = []
        if DOCUMENT_SRC_DOCUMENTS in self.index_sources:
            documents.extend(self.sqlite_adapter.get_docs())
        if DOCUMENT_SRC_EXAMPLES in self.index_sources:
            documents.extend(self.sqlite_adapter.get_examples())

        if self.max_docs:                       # smoke/pre-flight: tiny corpus
            n = len(documents)
            idxs = list(range(min(self.max_docs, n)))
            if self.max_docs_tail:              # also take the tail (e.g. junk docs)
                idxs += list(range(max(0, n - self.max_docs_tail), n))
            documents = [documents[i] for i in sorted(set(idxs))]
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
