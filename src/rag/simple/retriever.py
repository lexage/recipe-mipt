from typing import List, Tuple

from src.agent_constructor.core import Chunk, Document
from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.db import IDB


class SimpleRetriever(Retriever):
    def __init__(self, name: str, data_base: IDB):
        super().__init__(name)
        self.data_base = data_base

    def retrieve(self, query: str, k: int = 1) -> List[Chunk]:
        
        chunks = self.data_base.query(query_text=query, top_k=k)

        return chunks
