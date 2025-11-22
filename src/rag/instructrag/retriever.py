from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.agent import Agent
from src.agent_constructor.db import IDB


class InstructRAGRetriever(Retriever):
    def __init__(self, name, data_base: IDB, rationality_agent: Agent):
        super().__init__(name)
        self.data_base = data_base
        self.rationality_agent = rationality_agent

    def retrieve(self, query, k = 5):
        chunks = self.data_base.query(query, k)
        # Тут костыль, убрать, когда будут ContextAssemblers
        return [(self.rationality_agent.run(query, chunk.text), [chunk]) for chunk in chunks]
