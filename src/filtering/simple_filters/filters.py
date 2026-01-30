from typing import List

from src.agent_constructor.filters import Filter
from src.agent_constructor.chunkers import Chunk


class LengthFilter(Filter):
    def __init__(self, required: bool = False, min_len: int = 20):
        self.required = required
        self.min_len = min_len

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        result = [chunk for chunk in chunks if len(chunk.text.strip()) < self.min_len]
        return result
