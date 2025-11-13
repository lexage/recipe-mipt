from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.context_engine import CoRAGContextAssembler

class SimplePipeline(Pipeline):
    def __init__(self, retriever: Retriever, agent: Agent, context_assembler: ContextAssembler):
        super().__init__("simple_pipeline")
        self.retriever = retriever
        self.agent = agent
        self.context_assembler = context_assembler
            
    def run(self, task: str) -> str:
        context = self.retriever.retrieve(query=task)
        assembeld_context = self.context_assembler.assemble(context)
        return self.agent.run(task, assembeld_context)
