from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline


class BasePipeline(Pipeline):

    def __init__(
        self,
        agent: Agent
    ):

        super().__init__("simple_pipeline")

        self.agent = agent


    def run(self, task: str):
        final_answer = self.agent.run(task)
        final_answer = final_answer.replace("```py\n", "```python") # добавлено дополнительно изменение, которое фиксит проблему с получением пустых строк из ```py
        final_answer = final_answer.replace("```\n", "") # добавлено дополнительно изменение, которое фиксит проблему с получением пустых строк из ```
        return final_answer