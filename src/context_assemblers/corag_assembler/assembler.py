from typing import List

from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.core import Chunk

class CoRAGContextAssembler(ContextAssembler):
    def assemble(self, chunks: List[Chunk]):
        doc_sections = []
        inter_steps = ""

        for chunk in chunks:
            if chunk.id == "corag_intermediate_steps":
                inter_steps = chunk.text
            else:
                doc_sections.append(chunk.text.strip())

        if not doc_sections:
            return inter_steps

        docs = "# Relevant API Reference\n\n" + "\n\n---\n\n".join(doc_sections)
        return docs + ("\n\n" + inter_steps if inter_steps else "")
