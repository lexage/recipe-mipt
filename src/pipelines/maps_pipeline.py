from __future__ import annotations

import random

from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List,
    Optional,
    Tuple,
)

from agent_constructor.core import Text
from agent_constructor.agent import Agent

class MAPSAgentNames(Enum):
    ALIGNER = 'aligner'
    SCHOLAR = 'scholar'
    SOLVER = 'solver'
    USER_PROXY = 'user_proxy'
    CRITIC = 'critic'
    MANAGER = 'manager'


# ---------- Agent primitives: Planner, Critic, Student ----------


class Aligner(Agent):
    """Alignes the caption, context, and question to ensure the safe integration of these elements."""

    def __init__(self, name: str = "Aligner"):
        super().__init__(name)

    def run(self, task_description: Text, context: Text, feedback: Text) -> Text:
        return f"Aligned task description '{task_description}' based on '{feedback}' feedback."


class Scholar(Agent):
    """Researches the professional knowledge required by problems and exploring various hypotheses"""

    def __init__(self, name: str = "Scholar"):
        super().__init__(name)

    def run(self, task_description: Text, aligned_info: Text, context: Text, feedback: Text) -> Text:
        return f"Research results for task description '{task_description}' based on '{feedback}' feedback."


class Solver(Agent):
    """Gatheres all necessary information and resolving MSPs by selecting the most appropriate experimental approach"""

    def __init__(self, name: str = "Solver"):
        super().__init__(name)

    def run(self, task_description: Text, aligned_info: Text, research: Text, feedback: Text) -> List[Text]:
        return f"Solution for task description '{task_description}' based on '{feedback}' feedback."


class Critic(Agent):
    """Provides feedback and continuous correction throughout the solving process"""

    def __init__(self, name: str = "Critic"):
        super().__init__(name)

    def run(self, solution: Text, research: Text, aligned_info: Text) -> Tuple[List[int], List[str]]:
        return [random.randint(1, 5) for _ in range(4)], ["random feedback" for _ in range(4)]


class Manager(Agent):
    """Creates the experimental plan and schedule, ensuring each step is executed according to the predefined plan."""

    def __init__(self, name: str = "Manager"):
        super().__init__(name)

    def run(self, state: MAPSPiplineState) -> List[MAPSAgentNames]:
        return [MAPSAgentNames.ALIGNER, MAPSAgentNames.SCHOLAR, MAPSAgentNames.SOLVER]


class UserProxy(Agent):
    """Ensures smooth information flow and coordinating the allocation of tasks within the experiment"""

    def __init__(self, name: str = "User Proxy"):
        super().__init__(name)

    def run(self, task: Text, context: Text) -> Text:
        return f"Describtion of task '{task}'"

# ---------- Orchestration: Pipeline & Workflows ----------


@dataclass
class MAPSPipelineConfig:

    aligner: Aligner
    scholar: Scholar
    solver: Solver
    critic: Critic
    manager: Manager
    user_proxy: UserProxy
    max_iterations: int = 3
    loop_until_accepted: bool = True


@dataclass
class MAPSPiplineState:
    diagram: str
    context: str
    question: str

    aligned_info: Optional[str] = None
    research: Optional[str] = None
    solution: Optional[str] = None

    scores: Optional[List[int]] = field(default_factory=lambda: [-1] * 3)
    feedback: Optional[List[str]] = field(default_factory=lambda: [""] * 3)


class MAPSPipeline:

    def __init__(self, cfg: MAPSPipelineConfig):
        self.cfg = cfg

    def run(self, task: Text) -> Text:

        context = ""
        itteration = 0

        state = MAPSPiplineState(
            diagram=task,
            context=context,
            question=task
        )

        task_description = self.cfg.user_proxy.run(
            state.question, state.context)

        for itteration in range(self.cfg.max_iterations):

            plan = self.cfg.manager.run(state)
            for step in plan:

                if step == MAPSAgentNames.ALIGNER:
                    state.aligned_info = self.cfg.aligner.run(
                        task_description,
                        state.context,
                        state.feedback[0]
                    )

                if step == MAPSAgentNames.SCHOLAR:
                    state.research = self.cfg.scholar.run(
                        task_description, state.aligned_info, state.context, state.feedback[1])

                if step == MAPSAgentNames.SOLVER:
                    state.solution = self.cfg.solver.run(
                        task_description, state.aligned_info, state.research, state.feedback[2])

            scores, feedback = self.cfg.critic.run(
                state.solution, state.research, state.aligned_info)

            if min(scores) >= 5:
                break
            state.scores = scores
            state.feedback = feedback

        return state.solution

if __name__ == "__main__":
    
    aligner = Aligner()
    scholar = Scholar()
    solver = Solver()
    critic = Critic()
    manager = Manager()
    user_proxy = UserProxy()

    config = MAPSPipelineConfig(
        aligner=aligner,
        scholar=scholar,
        solver=solver,
        critic=critic,
        manager=manager,
        user_proxy=user_proxy,
    )

    pipline = MAPSPipeline(cfg=config)
    print(pipline.run("test task"))
