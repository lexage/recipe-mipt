from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.core import Text


class REWOOPipeline(Pipeline):

    def __init__(self, planner: Agent, worker: Agent, solver: Agent):
        super().__init__("rewoo_pipeline")
        self.planer = planner
        self.worker = worker
        self.solver = solver

    def run(self, task: Text):
        plan = self.planer.run(task)
        evidencies = self.worker.run(plan)
        final_answer = self.solver.run(task, plan, evidencies)
        return final_answer
