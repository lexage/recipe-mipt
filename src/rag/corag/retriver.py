from src.agents.agent_constructor.context_engine import Retriever, Chunk
from typing import List, Tuple

class CORAG_Retriver(Retriever):
    def __init__(self, name, data_base, planner_agent):
        super().__init__(name)
        self.data_base = data_base
        self.planner_agent = planner_agent

    def retrieve(self, query:str, k:int = 5) -> List[Tuple]:
        sub_queries = self.planner_agent.run(query)
        results = self.data_base.search(sub_queries, k)
        return results


