from abc import ABC, abstractmethod
from typing import Optional, List

from src.agent_constructor.core import Block, Chunk, Document


class Filter(Block):
    """Decide whether a chunk should be kept, or transform it.

    Filters can be optional or required per source. They operate on Chunks
    (post-chunking).  For document-level (pre-chunking) filtering see
    `DocumentFilter` below.
    """

    required: bool = False

    @abstractmethod
    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        """Apply filtering or transformation to a list of chunks.

        This method processes a list of Chunk objects and returns a new list
        containing only the chunks that should be retained, possibly after
        modification. To drop a chunk, simply exclude it from the returned list.

        Args:
            chunk (List[Chunk]): A list of Chunk instances to process.

        Returns:
            List[Chunk]: A list of processed chunks to keep.
        """
        raise NotImplementedError


class DocumentFilter(Block):
    """Decide whether a Document should be kept, or transform it.

    Counterpart of `Filter` but operates on whole Documents BEFORE chunking.
    Used by pipelines that support an extra `document_filter` step (e.g.
    `SimplePipelineWithDocFilter`) — the base `SimplePipeline` does not call
    document filters.
    """

    required: bool = False

    @abstractmethod
    def apply(self, documents: List[Document]) -> List[Document]:
        """Filter / transform a list of Documents.

        Args:
            documents: list of Document instances to process.

        Returns:
            List[Document]: subset / transformed list to keep.
        """
        raise NotImplementedError
