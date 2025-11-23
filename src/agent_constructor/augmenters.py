from abc import ABC, abstractmethod
from typing import (
    List,
)

from core import Block, Chunk

class Augmenter(Block):
    """Produce augmented chunks for a given chunk (paraphrases, synthetic examples, topics...)."""

    @abstractmethod
    def augment(self, chunk: Chunk) -> List[Chunk]:
        raise NotImplementedError