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
        plan: Text = self.planner.run(task)

        steps: List[Text] = [
            line for line in plan.splitlines() if line.strip()
        ]

        evidences: List[Text] = asyncio.run(self._worker_run(task, steps))

        steps_complitions = [
            f"Plan {i + 1}: {p}\nEvidence {i + 1}: {e}"
            for i, (p, e) in enumerate(zip(steps, evidences))
        ]
        plan_complition: Text = "\n".join(steps_complitions)

        final_answer: Text = self.solver.run(
            task,
            context=plan_complition,
        )
        return final_answer

    async def _worker_run(self, task: Text, steps: List[Text]) -> List[Text]:
        worker_tasks = [
            asyncio.create_task(self.worker.run(step, context=task))
            for step in steps
        ]
        evidences = await asyncio.gather(*worker_tasks)
        return list(evidences)
