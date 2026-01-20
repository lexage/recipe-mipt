from abc import ABC, abstractmethod
from typing import (
    List,
)
import uuid

from src.agent_constructor.core import Block, Document, Chunk


class Chunker(Block):
    """Break a Document into a sequence of Chunks.

    Different sources may use different chunkers. Implementations should be
    deterministic and fast.
    """

    @abstractmethod
    def chunk(self, doc: Document) -> List[Chunk]:
        raise NotImplementedError


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


class DummyChunker(Chunker):
    def __init__(self):
        pass

    def chunk(self, doc: Document) -> List[Chunk]:
        text = doc.text
        chunk = Chunk(id=str(doc.id), doc_id=doc.id, text=text, tokens=None, metadata=doc.metadata)
        return [chunk]
    
    
class DSIRChunker(Chunker):
    def __init__(self, chunk_length: int = 128):
        self.chunk_length = chunk_length
    
    def chunk(self, doc: Document) -> List[Chunk]:
        chunks = []
        words = doc.text.split(' ')
        
        text_chunks = [' '.join(words[i:i + self.chunk_length]) 
                       for i in range(0, len(words), self.chunk_length)]
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                id=f"{doc.id}_chunk_{i}",
                doc_id=doc.id,
                text=chunk_text,
                tokens=None,
                metadata=doc.metadata.copy()
            )
            chunks.append(chunk)
        
        return chunks
