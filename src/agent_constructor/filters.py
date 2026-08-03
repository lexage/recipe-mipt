from abc import ABC, abstractmethod
from typing import Optional, List

from src.agent_constructor.core import Block, Chunk


class Filter(Block):
    """Decide whether a chunk/document should be kept, or transform it.

    Filters can be optional or required per source. They operate on Chunks.
    """

    required: bool = False

    @abstractmethod
    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        """Apply filtering or transformation to a list of chunks.

        This method processes a list of Chunk objects and returns a new list
        containing only the chunks that should be retained, possibly after
        modification. To drop a chunk, simply exclude it from the returned list.

        Args:
            chunk (List[Chunk]): A list of Chunk instances to process.

        Returns:
            List[Chunk]: A list of processed chunks to keep.
        """
        raise NotImplementedError
