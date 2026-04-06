import logging

from src.agent_constructor.core import Text
from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline

_LOG_SEPARATOR = f"\n{'_' * 20}\n"


class MARSPipeline(Pipeline):

    def __init__(
        self,
        planner: Agent,
        teacher: Agent,
        critic: Agent,
        student: Agent,
        max_critic_attempts: int = 3,
    ):
        super().__init__("mars_pipeline")
        self.planner = planner
        self.teacher = teacher
        self.critic = critic
        self.student = student
        self.max_critic_attempts = max_critic_attempts

    def run(self, task: Text) -> Text:

        logging.info(f"TASK: {task}")
        logging.info(_LOG_SEPARATOR)

        final_answer = self.student.run(task, mode="init")

        logging.info(f"INITIAL FINAL ANSWER: {final_answer}")
        logging.info(_LOG_SEPARATOR)

        steps = self.planner.run(task, final_answer)

        logging.info(f"STEPS: {steps}")
        logging.info(_LOG_SEPARATOR)

        for idx, step in enumerate(steps, start=1):
            score = False
            attempt = 0
            max_attempts = self.max_critic_attempts
            questions = None
            critic_feedback = None

            while not score and attempt < max_attempts:

                if critic_feedback:
                    questions = self.teacher.run(
                        task, step, final_answer, critic_feedback, mode="regenerate"
                    )
                    logging.info(f"REGENERATE QUESTIONS: {questions}")
                    logging.info(_LOG_SEPARATOR)
                else:
                    questions = self.teacher.run(task, step, final_answer, mode="ask")
                    logging.info(f"QUESTIONS: {questions}")
                    logging.info(_LOG_SEPARATOR)

                critic_response = self.critic.run(questions)

                logging.info(f"CRITIC: {critic_response}")
                logging.info(_LOG_SEPARATOR)

                if "False" in critic_response:
                    score = False
                    critic_feedback = critic_response
                else:
                    score = True

                attempt += 1

            if questions:
                final_answer = self.student.run(task, final_answer, questions)

                logging.info(f"FINAL ANSWER ON STEP {idx}: {final_answer}")
                logging.info(_LOG_SEPARATOR)

        logging.info(f"RESULT FINAL ANSWER: {final_answer}")
        logging.info(_LOG_SEPARATOR)

        return final_answer
