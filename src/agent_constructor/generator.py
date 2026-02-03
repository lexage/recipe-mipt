from abc import abstractmethod
from typing import Optional, List

from src.agent_constructor.core import Text
from src.agent_constructor.core import Block, Document


class Generator(Block):
    """Base class for synthetic document generators."""

    @abstractmethod
    def generate(self, documents: List[Document]) -> List[Document]:
        """Generate synthetic documents from input documents.

        Args:
            documents (List[Document]): Optional source documents used to guide generation.

        Returns:
            List[Document]: Generated synthetic documents.
        """
        raise NotImplementedError
