from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline

class SimplePipeline(Pipeline):
    def __init__(self, retriever: Retriever, agent: Agent):
        super().__init__("simple_pipeline")
        self.retriever = retriever
        self.agent = agent
            
    def run(self, task: str) -> str:
        context = self.retriever.retrieve(query=task)[0][1][0].text

        return self.agent.run(context, task)