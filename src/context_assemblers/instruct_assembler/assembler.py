from typing import List, Dict

from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.core import Chunk


class InstructRAGContextAssembler(ContextAssembler):
    def assemble(self, data: List[Chunk]):
        
        original_chunks: Dict[str, Chunk] = {}
        rationalities_chunks: List[Chunk] = []

        for chunk in data:
            source = (chunk.metadata or {}).get("source")
            if source in ("instruct_rag", "api_instruct_rag"):
                rationalities_chunks.append(chunk)
            else:
                original_chunks[chunk.id] = chunk

        context = ""

        for idx, rationale_chunk in enumerate(rationalities_chunks):
            api = (rationale_chunk.metadata or {}).get("retrieval_api")
            if api:
                context += f"# API [{idx}]: {api}\n"

            doc = original_chunks.get(rationale_chunk.doc_id)
            if doc is not None:
                context += f"# Documentation [{idx}]:\n{doc.text}\n"

            context += f"# Rationale [{idx}]:\n{rationale_chunk.text}\n\n"

        return context.strip()
