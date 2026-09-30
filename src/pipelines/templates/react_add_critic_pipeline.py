from src.agent_constructor.agent import Agent
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.core import Text


class REACTPipelineAddCritic(Pipeline):
    def __init__(self, react_agent: Agent, critic_agent: Agent, solver_agent: Agent):
        super().__init__("react_pipeline")
        self.react_agent = react_agent
        self.critic_agent = critic_agent
        self.solver_agent = solver_agent

    def run(self, task: Text):
        react_answer = self.react_agent.run(task) # возвращает ответ на задачу
        critic_answer = self.critic_agent.run(task, react_answer) # возвращает критику на задачу
        final_answer = self.solver_agent.run(task, react_answer, critic_answer)
        
        final_answer = final_answer.replace("```py\n", "```python") # добавлено дополнительно изменение, которое фиксит проблему с получением пустых строк из ```py
        final_answer = final_answer.replace("```\n", "") # добавлено дополнительно изменение, которое фиксит проблему с получением пустых строк из ```
        return final_answer
