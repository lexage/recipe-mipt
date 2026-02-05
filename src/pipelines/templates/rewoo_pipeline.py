from typing import List
import asyncio

from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.core import Text


class REWOOPipeline(Pipeline):
    """
    ReWOO-style pipeline with three modules:
    Planner -> Worker -> Solver.

    - Planner: given the original task, produces an ordered list of textual plans.
    - Worker: for each plan step, retrieves/produces supporting evidence.
    - Solver: given the task, full plan and collected evidence, produces the final answer.
    """

    def __init__(self, planner: Agent, worker: Agent, solver: Agent):
        super().__init__("rewoo_pipeline")
        self.planner = planner
        self.worker = worker
        self.solver = solver

    def run(self, task: Text) -> Text:
        # 1) Planner composes a full blueprint for the task
        plan: List[Text] = self.planner.run(task)

        # 2) Worker collects evidence for each plan step (can be parallelised)
        evidences: List[Text] = asyncio.run(self._worker_run(task, plan))

        # 3) Solver combines task, plan and evidences into the final answer
        final_answer: Text = self.solver.run(task, plan, evidences)
        return final_answer

    async def _worker_run(self, task: Text, plan: List[Text]) -> List[Text]:
        worker_tasks = [
            asyncio.create_task(self.worker.run(task, sub_task))
            for sub_task in plan
        ]
        evidences = await asyncio.gather(*worker_tasks)
        return list(evidences)
