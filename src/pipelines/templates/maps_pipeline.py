from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum, auto

from src.agent_constructor.core import Text
from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.context_engine import Retriever


class MAPSAgents(Enum):
    ALIGNER = auto()
    SCHOLAR = auto()
    SOLVER = auto()
    CRITIC = auto()


@dataclass
class MAPSPipelineState:
    
    task: Text
    context: Text

    aligned_info: Text = ""
    research: Text = ""
    solution: Text = ""
    feedback: Text = ""

    scores: Optional[List[int]] = field(default_factory=lambda: [-1] * 3)


class MAPSPipeline(Pipeline):

    def __init__(self, aligner: Agent, scholar: Agent, solver: Agent, critic: Agent, retriever: Retriever = None, max_iterations: int = 1):
        super().__init__("maps_pipeline")
        self.retriever = retriever
        self.aligner = aligner 
        self.scholar = scholar 
        self.solver = solver
        self.critic = critic
        self.max_iterations = max_iterations

    def run(self, task: Text) -> Text:
        
        context = ""
        if self.retriever:
            context = self.retriever.retrieve(task)
        
        state = MAPSPipelineState(
            task=task,
            context=context,
        )
        
        next_agent = MAPSAgents.ALIGNER
        i = 0
        while i < self.max_iterations:

            match next_agent:

                case MAPSAgents.ALIGNER:

                    state.aligned_info = self.aligner.run(
                    state.task, 
                    state.context + state.feedback
                    )

                    next_agent = MAPSAgents.SCHOLAR

                case MAPSAgents.SCHOLAR:

                    state.research = self.scholar.run(
                    state.task, 
                    state.aligned_info + state.context + state.feedback
                    )

                    next_agent = MAPSAgents.SOLVER

                case MAPSAgents.SOLVER:

                    state.solution = self.solver.run(
                    state.task, 
                    state.research + state.aligned_info + state.context + state.feedback)

                    next_agent = MAPSAgents.CRITIC

                case MAPSAgents.CRITIC:

                    state.scores, state.feedback = self.critic.run(
                        self._critic_prompt(
                            state.aligned_info, 
                            state.research,
                            state.solution
                        )
                    )

                    if min(state.scores) >= 5:
                        break
                    
                    agent_id = state.scores.index(min(state.scores))
                    next_agent = [MAPSAgents.ALIGNER, MAPSAgents.SCHOLAR, MAPSAgents.SOLVER][agent_id]
                    i += 1

        return state.solution

    def _critic_prompt(
            self, 
            aligner_results: Text, 
            scholar_results: Text,
            solver_results: Text
            ):
        
        return f"""
[alignment]: \n{aligner_results}\n\n
[knowledge]: \n{scholar_results}\n\n
[solution]: \n{solver_results}\n\n
"""
