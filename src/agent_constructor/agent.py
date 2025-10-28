from abc import abstractmethod
from src.agent_constructor.core import Block


class Agent(Block):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

