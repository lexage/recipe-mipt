from typing import List

from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.agent import Agent
from src.agent_constructor.db import IDB
from src.agent_constructor.core import Text, Chunk, Document


class InstructRAGRetriever(Retriever):
    def __init__(
            self, 
            name, 
            data_base: IDB, 
            rationality_agent: Agent,
            documents_as_chunks: bool = False
            ):
        super().__init__(name)
        self.data_base = data_base
        self.rationality_agent = rationality_agent
        self.documents_as_chunks = documents_as_chunks

    def retrieve(self, query: Text, k: int = 5) -> List[Chunk]:
        chunks = self.data_base.query(query, k)

        if self.documents_as_chunks:
            
            doc_ids = list(set([chunk.doc_id for chunk in chunks]))

            documnets: List[Document] = self.data_base.get_documents(
                ids=doc_ids
            )

            chunks = [
                Chunk(
                    id=doc.id,
                    doc_id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                )
                for doc in documnets
            ]

        rationalities = [self.rationality_agent.run(query, chunk.text) for chunk in chunks]

        chunks.extend(
            self._create_rationality_chunks(rationalities, chunks)
        )

        return chunks
    
    @staticmethod
    def _create_rationality_chunks(rationalities: List[Text], chunks: List[Chunk]) -> List[Chunk]:
        
        rationalities_chunks: List[Chunk] = []
        for idx, (rationality, chunk) in enumerate(zip(rationalities, chunks)):
            rationalities_chunks.append(
                Chunk(
                    id=f"instruct_{idx}",
                    doc_id=chunk.id,
                    text=rationality,
                    metadata={
                        "source": "instruct_rag"
                    }
                )
            )
        return rationalities_chunks
