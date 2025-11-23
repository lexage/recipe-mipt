from abc import ABC, abstractmethod
from typing import (
    List,
)
import uuid

from src.agent_constructor.core import Block, Document, Chunk
from src.pipelines.registry import register_component
from src.pipelines.configs import ChunkerConfig
from src.pipelines.constants import ComponentNames


class Chunker(Block):
    """Break a Document into a sequence of Chunks.

    Different sources may use different chunkers. Implementations should be
    deterministic and fast.
    """

    @abstractmethod
    def chunk(self, doc: Document) -> List[Chunk]:
        raise NotImplementedError


@register_component(ChunkerConfig, ComponentNames.SIMPLE_CHUNKER)
class SimpleChunker(Chunker):
    def __init__(self, max_chars: int = 1000):
        self.max_chars = max_chars

    def chunk(self, doc: Document) -> List[Chunk]:
        text = doc.text
        chunks: List[Chunk] = []
        i = 0
        while i < len(text):
            piece = text[i : i + self.max_chars]
            chunk = Chunk(id=str(uuid.uuid4()), doc_id=doc.id, text=piece, tokens=None, metadata={'doc_id': doc.id})
            chunks.append(chunk)
            i += self.max_chars
        return chunks


@register_component(ChunkerConfig, ComponentNames.DUMMY_CHUNKER)
class DummyChunker(Chunker):
    def __init__(self):
        pass

    def chunk(self, doc: Document) -> List[Chunk]:
        text = doc.text
        chunk = Chunk(id=str(doc.id), doc_id=doc.id, text=text, tokens=None)
        return [chunk]
