from typing import List
import asyncio

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
        evidencies = asyncio.run(self._worker_run(plan))
        final_answer = self.solver.run(plan, evidencies)
        return final_answer

    async def _worker_run(self, plan: List[Text]):
        worker_tasks = [asyncio.create_task(self.worker.run(sub_task)) for sub_task in plan]
        evidencies = await asyncio.gather(*worker_tasks)
        return evidencies
