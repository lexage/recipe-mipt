from abc import abstractmethod
from typing import Any

from src.agent_constructor.core import Block


class BaseTool(Block):
    """Абстрактный класс инструмента"""

    def __init__(self, name: str = "", description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass
