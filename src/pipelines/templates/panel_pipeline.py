from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.core import Text


class PANELPipeline(Pipeline):
    def __init__(self, agent: Agent):
        super().__init__("panel_pipeline")
        self.agent = agent

    def run(self, task: Text):
        final_answer = self.agent.run(task)
        return final_answer
