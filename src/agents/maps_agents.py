from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
import random

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.pipelines.registry import register_component
from src.pipelines.configs import AgentConfig
from src.pipelines.constants import ComponentNames


class MAPSAgentNames(Enum):
    ALIGNER = 'aligner'
    SCHOLAR = 'scholar'
    SOLVER = 'solver'
    USER_PROXY = 'user_proxy'
    CRITIC = 'critic'
    MANAGER = 'manager'


@dataclass
class MAPSPipelineState:
    diagram: str
    context: str
    question: str

    aligned_info: Optional[str] = None
    research: Optional[str] = None
    solution: Optional[str] = None

    scores: Optional[List[int]] = field(default_factory=lambda: [-1] * 3)
    feedback: Optional[List[str]] = field(default_factory=lambda: [""] * 3)

# Суда я хочу встроить такю херню, что если мы указываем имя модели и есть 
# подключение к хочут вллм, то испуользуем вызов LLM, если нет, то юзаем дамми
@register_component(AgentConfig, ComponentNames.MAPS_ALIGNER)
class AlignerMAPS(Agent):
    """Alignes the caption, context, and question to ensure the safe integration of these elements."""

    def __init__(self, name: str = "maps_aligner"):
        super().__init__(name)

    def run(self, task_description: Text, context: Text, feedback: Text) -> Text:
        return f"Aligned task description '{task_description}' based on '{feedback}' feedback."


@register_component(AgentConfig, ComponentNames.MAPS_SCHOLAR)
class ScholarMAPS(Agent):
    """Researches the professional knowledge required by problems and exploring various hypotheses"""

    def __init__(self, name: str = "maps_scholar"):
        super().__init__(name)

    def run(self, task_description: Text, aligned_info: Text, context: Text, feedback: Text) -> Text:
        return f"Research results for task description '{task_description}' based on '{feedback}' feedback."

 
@register_component(AgentConfig, ComponentNames.MAPS_SOLVER)
class SolverMAPS(Agent):
    """Gatheres all necessary information and resolving MSPs by selecting the most appropriate experimental approach"""

    def __init__(self, name: str = "maps_solver"):
        super().__init__(name)

    def run(self, task_description: Text, aligned_info: Text, research: Text, feedback: Text) -> List[Text]:
        return f"Solution for task description '{task_description}' based on '{feedback}' feedback."


@register_component(AgentConfig, ComponentNames.MAPS_CRITIC)
class CriticMAPS(Agent):
    """Provides feedback and continuous correction throughout the solving process"""

    def __init__(self, name: str = "maps_critic"):
        super().__init__(name)

    def run(self, solution: Text, research: Text, aligned_info: Text) -> Tuple[List[int], List[str]]:
        return [random.randint(1, 5) for _ in range(4)], ["random feedback" for _ in range(4)]


@register_component(AgentConfig, ComponentNames.MAPS_MANAGER)
class ManagerMAPS(Agent):
    """Creates the experimental plan and schedule, ensuring each step is executed according to the predefined plan."""

    def __init__(self, name: str = "maps_manager"):
        super().__init__(name)

    def run(self, state: MAPSPipelineState) -> List[MAPSAgentNames]:
        return [MAPSAgentNames.ALIGNER, MAPSAgentNames.SCHOLAR, MAPSAgentNames.SOLVER]


@register_component(AgentConfig, ComponentNames.MAPS_USER_PROXY)
class UserProxyMAPS(Agent):
    """Ensures smooth information flow and coordinating the allocation of tasks within the experiment"""

    def __init__(self, name: str = "user_proxy"):
        super().__init__(name)

    def run(self, task: Text, context: Text) -> Text:
        return f"Description of task '{task}'"