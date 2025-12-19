from src.agent_constructor.core import Text
from src.agents.pipelines.maps import MAPSAgentNames, MAPSPipelineState
from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline

class MAPSPipeline(Pipeline):

    def __init__(self, user_proxy: Agent, aligner: Agent, scholar: Agent, solver: Agent, critic: Agent, manager: Agent, max_iterations: int):
        super().__init__("maps_pipeline")
        self.user_proxy = user_proxy 
        self.aligner = aligner 
        self.scholar = scholar 
        self.solver = solver 
        self.critic = critic 
        self.manager = manager 
        self.max_iterations = max_iterations

    def run(self, task: Text) -> Text:

        context = ""

        state = MAPSPipelineState(
            diagram=task,
            context=context,
            question=task
        )

        task_description = self.user_proxy.run(
            state.question, state.context)

        for _ in range(self.max_iterations):

            plan = self.manager.run(state)
            for step in plan:

                if step == MAPSAgentNames.ALIGNER:
                    state.aligned_info = self.aligner.run(
                        task_description,
                        state.context,
                        state.feedback[0]
                    )

                if step == MAPSAgentNames.SCHOLAR:
                    state.research = self.scholar.run(
                        task_description, state.aligned_info, state.context, state.feedback[1])

                if step == MAPSAgentNames.SOLVER:
                    state.solution = self.solver.run(
                        task_description, state.aligned_info, state.research, state.feedback[2])

            scores, feedback = self.critic.run(
                state.solution, state.research, state.aligned_info)

            if min(scores) >= 5:
                break
            state.scores = scores
            state.feedback = feedback

        return state.solution
