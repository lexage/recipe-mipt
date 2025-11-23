from typing import List

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.pipelines.registry import register_component
from src.pipelines.configs import AgentConfig
from src.pipelines.constants import ComponentNames


@register_component(AgentConfig, ComponentNames.REWOO_PLANNER)
class PlannerREWOO(Agent):
    def __init__(self, name: str = "rewoo_planner_agent", maximum_steps: int = 5):
        super().__init__(name)
        self.maximum_steps = maximum_steps
    
    def run(self, task: Text) -> List[Text]:
        return [f"#{i} Sub-Task for Task '{task}'" for i in range(self.maximum_steps)]


@register_component(AgentConfig, ComponentNames.REWOO_WORKER)
class WorkerREWOO(Agent):
    def __init__(self, name: str = "rewoo_worker_agent"):
        super().__init__(name)
    
    async def run(self, task: Text) -> Text:
        return f"Evidence for '{task}'"


@register_component(AgentConfig, ComponentNames.REWOO_SOLVER)
class SolverREWOO(Agent):
    def __init__(self, name: str = "rewoo_solver_agent"):
        super().__init__(name)

    def run(self, plan: List[Text], evidencies: List[Text]):
        final_answer = "FINAL ANSWER FOR:\n"+'\n\n'.join([f"\t- Plan: '{p}'\n\t- Evidence: '{e}'" for p,e in zip(plan, evidencies)])
        return final_answer
