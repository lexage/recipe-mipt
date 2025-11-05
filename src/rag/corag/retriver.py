from typing import List, Tuple

from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.agent import Agent
from src.agent_constructor.db import IDB


class CoRAGRetriver(Retriever):
    def __init__(self, name: str, data_base: IDB, planner_agent: Agent):
        super().__init__(name)
        self.data_base = data_base
        self.planner_agent = planner_agent

    def retrieve(self, query: str, k:int = 5) -> List[Tuple]:
        sub_queries = self.planner_agent.run(query)
        results = [self.data_base.query(sub_query, k) for sub_query in sub_queries]
        return [(sub_query, chunks) for sub_query, chunks in zip(sub_queries, results)]

