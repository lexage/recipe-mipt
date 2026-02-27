from abc import abstractmethod
from typing import (
    List
)

from src.agent_constructor.core import Block, Chunk

class ICLBlock(Block):
    """
    ICLBlock 
    """

    @abstractmethod
    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Return enchanced list of chunks
        """
        raise NotImplementedError