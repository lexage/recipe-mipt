from typing import List

from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.core import Chunk

class CoRAGContextAssembler(ContextAssembler):
    def assemble(self, chunks: List[Chunk]):
        
        documnets = "## Documents:\n"
        inter_steps = ""

        for chunk in chunks:
            if chunk.id == "corag_intermediate_steps":
                inter_steps = chunk.text
            else:
                documnets+=f"""### Doc {chunk.id}:\n{chunk.text.strip()}\n\n"""

        return documnets + "\n" + inter_steps
