from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.core import Text


class REWOOPipelineAddCriticBeforeSolver(Pipeline):

    def __init__(self, planner: Agent, worker: Agent, solver: Agent, critic: Agent):
        super().__init__("rewoo_pipeline")
        self.planner = planner
        self.worker = worker
        self.solver = solver
        self.critic = critic

    def run(self, task: Text):
        plan = self.planner.run(task)
        evidencies = self.worker.run(plan)

        critic = self.critic.run(task, str(evidencies))

        final_answer = self.solver.run_after_critic(task, plan, evidencies, critic)
        final_answer = final_answer.replace("```py\n", "```python") # добавлено дополнительно изменение, которое фиксит проблему с получением пустых строк из ```py
        final_answer = final_answer.replace("```\n", "") # добавлено дополнительно изменение, которое фиксит проблему с получением пустых строк из ```
        
        return final_answer
    
class REWOOPipelineAddCriticAfterSolver(Pipeline):

    def __init__(self, planner: Agent, worker: Agent, solver: Agent, critic: Agent):
        super().__init__("rewoo_pipeline")
        self.planner = planner
        self.worker = worker
        self.solver = solver
        self.critic = critic

    def run(self, task: Text):
        plan = self.planner.run(task)
        evidencies = self.worker.run(plan)
        pre_final_answer = self.solver.run(task, plan, evidencies)
        critic = self.critic.run(task, pre_final_answer)
        final_answer = self.solver.run_after_critic(task, plan, evidencies, critic)
        final_answer = final_answer.replace("```py\n", "```python") # добавлено дополнительно изменение, которое фиксит проблему с получением пустых строк из ```py
        final_answer = final_answer.replace("```\n", "") # добавлено дополнительно изменение, которое фиксит проблему с получением пустых строк из ```
        return final_answer

