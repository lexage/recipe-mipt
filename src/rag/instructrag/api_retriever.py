from typing import List, Optional

from src.agent_constructor.agent import Agent
from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.core import Chunk, Text
from src.agent_constructor.db import IDB
from src.benchmarks import DataItemDS1000
from src.rag.api.retriever import APIRetriever


class APIInstructRAGRetriever(Retriever):
    """
    API-aware InstructRAG: select APIs for the task, retrieve doc chunks per API,
    then synthesize a rationale per chunk (usefulness + how to use, no solution).
    """

    def __init__(
        self,
        name: str,
        data_base: IDB,
        api_selector: Agent,
        rationality_agent: Agent,
        max_apis: int = 8,
        api_top_k: int = 4,
        chunks_per_api: int = 1,
        include_original_docs: bool = True,
        ignore_libs: Optional[List[str]] = None,
    ):
        super().__init__(name)
        self._api_retriever = APIRetriever(
            name=f"{name}_api",
            data_base=data_base,
            api_selector=api_selector,
            max_apis=max_apis,
            api_top_k=api_top_k,
            chunks_per_api=chunks_per_api,
            ignore_libs=ignore_libs,
        )
        self.rationality_agent = rationality_agent
        self.include_original_docs = include_original_docs

    def retrieve(self, query: DataItemDS1000 | Text, k: int = 6) -> List[Chunk]:
        prompt, _ = APIRetriever._extract_prompt_and_metadata(query)
        doc_chunks = self._api_retriever.collect_api_chunks(query=query, k=k)
        rationale_chunks = self._create_rationale_chunks(prompt, doc_chunks)

        if self.include_original_docs:
            return doc_chunks + rationale_chunks
        return rationale_chunks

    def _create_rationale_chunks(
        self,
        task: Text,
        doc_chunks: List[Chunk],
    ) -> List[Chunk]:
        rationale_chunks: List[Chunk] = []

        for idx, chunk in enumerate(doc_chunks):
            api = (chunk.metadata or {}).get("retrieval_api", "unknown")
            rationale = self.rationality_agent.run(
                task=task,
                api=api,
                documentation=chunk.text,
            )

            rationale_chunks.append(
                Chunk(
                    id=f"api_instruct_{idx}",
                    doc_id=chunk.id,
                    text=rationale,
                    metadata={
                        "source": "api_instruct_rag",
                        "retrieval_api": api,
                    },
                )
            )

        return rationale_chunks
