from dataclasses import dataclass
from typing import List
import asyncio

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text


class PlannerREWOO(Agent):
    def __init__(self, name: str = "REWOO_Planner", maximum_steps: int = 5):
        super().__init__(name)
        self.maximum_steps = maximum_steps
    
    def run(self, task: Text) -> List[Text]:
        return [f"#{i} Sub-Task for Task '{task}'" for i in range(self.maximum_steps)]


class WorkerREWOO(Agent):
    def __init__(self, name: str = "REWOO_Worker"):
        super().__init__(name)
    
    async def run(self, task: Text) -> Text:
        return f"Evidence for '{task}'"

class SolverREWOO(Agent):
    def __init__(self, name: str = "REWOO_Solver"):
        super().__init__(name)

    def run(self, plan: List[Text], evidencies: List[Text]):
        final_answer = "FINAL ANSWER FOR:\n"+'\n\n'.join([f"\t- Plan: '{p}'\n\t- Evidence: '{e}'" for p,e in zip(plan, evidencies)])
        return final_answer


@dataclass
class ConfigPiplineREWOO:
    planer: PlannerREWOO
    worker: WorkerREWOO
    solver: SolverREWOO


class PiplineREWOO:
    def __init__(self, config: ConfigPiplineREWOO):
        self.planer = config.planer
        self.worker = config.worker
        self.solver = config.solver

    def run(self, task: Text):
        plan = self.planer.run(task)
        evidencies = asyncio.run(self._worker_run(plan))
        final_answer = self.solver.run(plan, evidencies)
        return final_answer

    async def _worker_run(self, plan: List[Text]):
        worker_tasks = [asyncio.create_task(self.worker.run(sub_task)) for sub_task in plan]
        evidencies = await asyncio.gather(*worker_tasks)
        return evidencies


if __name__ == "__main__":

    planner = PlannerREWOO()
    solver = SolverREWOO()
    worker = WorkerREWOO()

    config = ConfigPiplineREWOO(
        planer=planner,
        solver=solver,
        worker=worker
    )

    pipline = PiplineREWOO(
        config=config
    )

    results = pipline.run("Test Task")

    print(results)
