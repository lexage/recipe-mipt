from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.db import IDB
from src.agent_constructor.chunkers import Chunker
from src.agent_constructor.filters import Filter
from src.agent_constructor.agent import Agent
from src.agent_constructor.generator import Generator
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.icl import ICLBlock


class SimplePipeline(Pipeline):

    def __init__(
        self,
        agent: Agent
    ):

        super().__init__("simple_pipeline")

        self.agent = agent


    def run(self, task: str) -> str:

        return self.agent.run(task, context)
