from typing import List, Tuple

from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.agent import Agent
from src.agent_constructor.db import IDB
from src.pipelines.registry import ComponentRegistry
from src.pipelines.constants import ComponentNames


@ComponentRegistry.register_component(ComponentNames.CORAG_RETRIEVER)
class CoRAGRetriever(Retriever):
    def __init__(self, name: str, data_base: IDB, generator: Agent, sub_solver: Agent, max_sub_queries: int):
        super().__init__(name)
        self.data_base = data_base
        self.generator = generator
        self.sub_solver = sub_solver
        self.max_sub_queries = max_sub_queries

    def retrieve(self, query: str, k: int = 1) -> List[Tuple]:
        prev_qna = []
        retrived_chunks = []
        for i in range(self.max_sub_queries):
            sub_query = self.generator.run(query, prev_qna)
            context = self.data_base.query(sub_query, k)
            sub_answer = self.sub_solver.run(sub_query, context)
            prev_qna.append((sub_query, sub_answer))
            retrived_chunks.extend(context)
        return prev_qna, retrived_chunks
