import logging

from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline


_LOG_SEPARATOR = f"\n{'_' * 20}\n"


class BasePipelineDS1000(Pipeline):

    def __init__(
        self,
        agent: Agent
    ):

        super().__init__("simple_pipeline")

        self.agent = agent


    def run(self, task: str):
        logging.info(f"TASK: {task}")
        logging.info(_LOG_SEPARATOR)

        final_answer = self.agent.run(task)
        
        logging.info(f"FINAL ANSWER BEFORE POSTPROCESS: {final_answer}")
        logging.info(_LOG_SEPARATOR)
        
        final_answer = final_answer.replace("```py\n", "```python") # добавлено дополнительно изменение, которое фиксит проблему с получением пустых строк из ```py
        final_answer = final_answer.replace("```\n", "") # добавлено дополнительно изменение, которое фиксит проблему с получением пустых строк из ```
        
        logging.info(f"FINAL ANSWER AFTER POSTPROCESS: {final_answer}")
        logging.info(_LOG_SEPARATOR)
        return final_answer

 
class BasePipelineCodeMMLU(Pipeline):

    def __init__(
        self,
        agent: Agent
    ):

        super().__init__("simple_pipeline")

        self.agent = agent


    def run(self, task: str):
        logging.info(f"TASK: {task}")
        logging.info(_LOG_SEPARATOR)

        final_answer = self.agent.run(task)
        
        logging.info(f"FINAL ANSWER: {final_answer}")
        logging.info(_LOG_SEPARATOR)

        return final_answer