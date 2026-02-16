from abc import ABC, abstractmethod
from typing import (
    List,
)
import uuid

from src.agent_constructor.core import Block, Document, Chunk, Text
from src.utils.adapters import DOCUMENT_SRC_EXAMPLES

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


class RecursiveChunker(Chunker):
    def __init__(
            self, 
            max_chunk_size: int = 1000, 
            separators: list = ['\n\n', '\n', '.', ',', ' ']
            ):
        
        self.max_chunk_size = max_chunk_size
        self.separators = separators

    def chunk(self, doc: Document) -> List[Chunk]:
        if doc.source == DOCUMENT_SRC_EXAMPLES:
            return [Chunk(
                id=str(doc.id) + "_0",
                doc_id=doc.id,
                text=doc.text,
                metadata=doc.metadata
            )]

        text_splits = self._split(doc.text)
        text_splits = self._merge(text_splits)

        chunks = []

        for i, text in enumerate(text_splits):
            chunks.append(
                Chunk(
                    id=str(doc.id) + "_" + str(i),
                    doc_id=doc.id,
                    text=text,
                    metadata=doc.metadata
                )
            )

        return chunks
    
    def _split(self, text: Text, separator_lvl: int = 0) -> List[Text]:
        if len(text) < self.max_chunk_size:
            return [text]
        
        if separator_lvl >= len(self.separators):
            return []
        
        result = []
        
        splits = text.split(self.separators[separator_lvl])

        for split in splits:

            result.extend(self._split(split, separator_lvl+1))

        return result
    
    def _merge(self, text_splits: List[Text]) -> List[Text]:
        merged_splits = []
        current_split = ""

        for text in text_splits:

            if len(current_split) + len(text) < self.max_chunk_size:
                current_split += text
            else:
                merged_splits.append(current_split)
                current_split = text

        merged_splits.append(current_split)

        return merged_splits

