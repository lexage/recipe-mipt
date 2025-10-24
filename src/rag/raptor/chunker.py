from typing import List

from src.agents.agent_constructor.chunkers import Chunker
from src.agents.agent_constructor.core import Document, Chunk


class DummyChunker(Chunker):
    def __init__(self):
        pass

    def chunk(self, doc: Document) -> List[Chunk]:
        text = doc.text
        chunk = Chunk(id=str(doc.id), doc_id=doc.id, text=text, tokens=None)
        return [chunk]
