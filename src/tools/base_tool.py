from abc import ABC, abstractmethod


class AbstractTool(ABC):

    """Абстрактный класс тула"""

    def __init__(self):
        self.desctiption = ""

    @abstractmethod
    def run(self):
        pass
    
    def call_tool(self, tool_name: str, **tool_kwargs):
        return self.tools[tool_name](**tool_kwargs)