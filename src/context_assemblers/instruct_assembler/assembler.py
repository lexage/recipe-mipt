from typing import List, Dict

from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.core import Chunk


class InstructRAGContextAssembler(ContextAssembler):
    def assemble(self, data: List[Chunk]):
        
        original_chunks: Dict[str: Chunk] = {}
        rationalities_chunks: List[Chunk] = []

        for chunk in data:
            if chunk.metadata.get("source", None) == "instruct_rag":
                rationalities_chunks.append(chunk)
            else:
                original_chunks[chunk.id] = chunk
        
        context = ""

        for idx, chunk in enumerate(rationalities_chunks):
            context+=f"# Document [{idx}]:\n{original_chunks[chunk.doc_id].text}\n"
            context+=f"# Rationale [{idx}]:\n{chunk.text}\n\n"

        return context.strip()
