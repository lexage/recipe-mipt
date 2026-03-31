from typing import List

from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.core import Chunk


class ExamplesAssembler(ContextAssembler):
    
    def assemble(self, chunks: List[Chunk]):
        context = "## Usefull examples:\n\n"
        for i, chunk in enumerate(chunks):
            context += f"Example #{i}:\n```{chunk.text}```\n\n"
        return context
