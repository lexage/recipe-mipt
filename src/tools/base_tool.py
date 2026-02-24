from abc import abstractmethod
from src.agent_constructor.core import Block


class BaseTool(Block):

    """Абстрактный класс тула"""

    def __init__(self):
        self.desctiption = ""

    @abstractmethod
    def run(self):
        pass
    
    def call_tool(self, tool_name: str, **tool_kwargs):
        return self.tools[tool_name](**tool_kwargs)