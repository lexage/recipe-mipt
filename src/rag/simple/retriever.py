from typing import List, Tuple

from src.agent_constructor.core import Chunk, Document
from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.db import IDB


class SimpleRetriever(Retriever):
    def __init__(self, name: str, data_base: IDB, return_docs_as_chunks: bool = False):
        super().__init__(name)
        self.data_base = data_base
        self.return_docs_as_chunks = return_docs_as_chunks

    def retrieve(self, query: str, k: int = 1) -> List[Chunk]:
        
        chunks = self.data_base.query(query_text=query, top_k=k)

        if not self.return_docs_as_chunks:
            return chunks

        doc_ids = list(set([chunk.doc_id for chunk in chunks]))

        documents : List[Document] = self.data_base.get_documents(doc_ids)

        chunks = [
            Chunk(
                id=doc.id,
                doc_id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
            ) 
            for doc in documents]

        return chunks

