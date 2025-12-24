from abc import ABC, abstractmethod
from typing import (
    Optional,
)

from src.agent_constructor.core import Block, Chunk

class Filter(Block):
    """Decide whether a chunk/document should be kept, or transform it.

    Filters can be optional or required per source. They operate on Chunks.
    """

    required: bool = False

    @abstractmethod
    def apply(self, chunk: Chunk) -> Optional[Chunk]:
        """Return the chunk (possibly modified) or None to drop it."""
        raise NotImplementedError


class LengthFilter(Filter):
    def __init__(self, required: bool = False, min_len: int = 20):
        self.required = required
        self.min_len = min_len

    def apply(self, chunk: Chunk) -> Optional[Chunk]:
        if len(chunk.text.strip()) < self.min_len:
            return None
        return chunk