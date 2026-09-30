import logging
from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.core import Text

# Константа для красивого разделения логов в консоли
_LOG_SEPARATOR = "=" * 40


class REWOOPipelineAddCriticBeforeSolver(Pipeline):

    def __init__(self, planner: Agent, worker: Agent, solver: Agent, critic: Agent):
        super().__init__("rewoo_pipeline")
        self.planner = planner
        self.worker = worker
        self.solver = solver
        self.critic = critic

    def run(self, task: Text):
        logging.info(f"START PIPELINE: REWOOPipelineAddCriticBeforeSolver")
        logging.info(f"TASK:\n{task}")
        logging.info(_LOG_SEPARATOR)

        logging.info("Running PLANNER...")
        plan = self.planner.run(task)
        logging.info(f"PLAN GENERATED:\n{plan}")
        logging.info(_LOG_SEPARATOR)

        logging.info("Running WORKER...")
        evidencies = self.worker.run(plan)
        logging.info(f"EVIDENCIES GATHERED:\n{evidencies}")
        logging.info(_LOG_SEPARATOR)

        logging.info("Running CRITIC (Before Solver)...")
        critic = self.critic.run(task, str(evidencies))
        logging.info(f"CRITIC FEEDBACK:\n{critic}")
        logging.info(_LOG_SEPARATOR)

        logging.info("Running SOLVER (After Critic)...")
        final_answer = self.solver.run(task, plan, evidencies, critic)
        # Фиксы форматирования кода
        final_answer = final_answer.replace("```py\n", "```python")
        final_answer = final_answer.replace("```\n", "")
        
        logging.info(f"PIPELINE FINAL ANSWER:\n{final_answer}")
        logging.info(_LOG_SEPARATOR)
        
        return final_answer


class REWOOPipelineAddCriticAfterSolver(Pipeline):

    def __init__(self, planner: Agent, worker: Agent, solver: Agent, critic: Agent):
        super().__init__("rewoo_pipeline")
        self.planner = planner
        self.worker = worker
        self.solver = solver
        self.critic = critic

    def run(self, task: Text):
        logging.info(f"START PIPELINE: REWOOPipelineAddCriticAfterSolver")
        logging.info(f"TASK:\n{task}")
        logging.info(_LOG_SEPARATOR)

        logging.info("Running PLANNER...")
        plan = self.planner.run(task)
        logging.info(f"PLAN GENERATED:\n{plan}")
        logging.info(_LOG_SEPARATOR)

        logging.info("Running WORKER...")
        evidencies = self.worker.run(plan)
        logging.info(f"EVIDENCIES GATHERED:\n{evidencies}")
        logging.info(_LOG_SEPARATOR)

        logging.info("Running SOLVER (Initial Pass)...")
        pre_final_answer = self.solver.run(task, plan, evidencies)
        logging.info(f"PRE-FINAL ANSWER:\n{pre_final_answer}")
        logging.info(_LOG_SEPARATOR)

        logging.info("Running CRITIC (After Solver)...")
        critic_feedback = self.critic.run(task, pre_final_answer)
        logging.info(f"CRITIC FEEDBACK:\n{critic_feedback}")
        logging.info(_LOG_SEPARATOR)

        logging.info("Running SOLVER (After Critic)...")
        final_answer = self.solver.run(task, plan, evidencies, critic_feedback=critic_feedback)
        
        # Фиксы форматирования кода
        final_answer = final_answer.replace("```py\n", "```python")
        final_answer = final_answer.replace("```\n", "")
        
        logging.info(f"PIPELINE FINAL ANSWER:\n{final_answer}")
        logging.info(_LOG_SEPARATOR)
        
        return final_answer