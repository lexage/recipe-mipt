import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import struct

from dataclasses import replace

from pydivsufsort import divsufsort, kasai

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Document, Chunk
from src.agent_constructor.filters import Filter


class ExactSubstrFiltrator(Filter):
    """Filter for removing exact duplicate substrings from text documents.
    
    Implements the ExactSubstr method described in the paper 
    'Deduplicating Training Data Makes Language Models Better'.
    """
    
    def __init__(
        self,
        threshold: int,
        enable_bytes: bool = True,
        enable_tokenizer: bool = False,
        tokenizer: Optional = None
    ) -> None:
        """Initialize the ExactSubstr filter.
        
        Args:
            threshold: Minimum length of duplicate substring to remove
            enable_bytes: Whether to work with byte-level representation
            enable_tokenizer: Whether to use tokenizer for encoding/decoding
            tokenizer: Tokenizer instance if enable_tokenizer is True
            
        Raises:
            NotImplementedError: If enable_bytes is False
            ValueError: If enable_tokenizer is True but no tokenizer provided
        """
        self.threshold = threshold
        self.enable_bytes = enable_bytes
        self.enable_tokenizer = enable_tokenizer
        self.tokenizer = tokenizer
        
        if not self.enable_bytes:
            raise NotImplementedError("This implementation requires enable_bytes=True")
        if self.enable_tokenizer and self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when enable_tokenizer=True")
    
    def _convert_documents_to_bytes(self, docs: List[str]) -> List[bytes]:
        """Convert documents to byte representation.
        
        Args:
            docs: List of document texts
            
        Returns:
            List of byte sequences representing the documents
        """
        encode_func = tok_encode if self.enable_tokenizer else encode
        return [encode_func(doc, self.tokenizer) if self.enable_tokenizer else encode_func(doc) for doc in docs]
    
    def _build_concatenated_string(self, doc_bytes_list: List[bytes]) -> Tuple[bytes, List[int], List[int]]:
        """Build concatenated byte string with document separators.
        
        Args:
            doc_bytes_list: List of document byte sequences
            
        Returns:
            Tuple containing:
            - Concatenated byte string with separators
            - List of starting offsets for each document
            - List of document lengths
        """
        separator = b'\x00'
        total_bytes = bytearray()
        offsets = []
        lengths = [len(b) for b in doc_bytes_list]
        
        for i, b in enumerate(doc_bytes_list):
            if i > 0:
                total_bytes.extend(separator)
            offsets.append(len(total_bytes))
            total_bytes.extend(b)
        
        return bytes(total_bytes), offsets, lengths
    
    def _create_document_map(self, total_bytes: bytes, offsets: List[int], lengths: List[int]) -> np.ndarray:
        """Create document ID map for each position in concatenated string.
        
        Args:
            total_bytes: Concatenated byte string
            offsets: Document starting positions
            lengths: Document lengths
            
        Returns:
            Numpy array mapping each position to its document ID
            (-1 for invalid positions, -2 for separators)
        """
        n_total = len(total_bytes)
        total_doc_ids = np.full(n_total, -1, dtype=np.int32)
        n_docs = len(offsets)
        
        for i in range(n_docs - 1):
            sep_pos = offsets[i] + lengths[i]
            if sep_pos < n_total:
                total_doc_ids[sep_pos] = -2
        
        for i, (start, length) in enumerate(zip(offsets, lengths)):
            end = min(start + length, n_total)
            total_doc_ids[start:end] = i
        
        return total_doc_ids
    
    def _find_duplicate_intervals(
        self,
        suffix_array: np.ndarray,
        lcp_array: np.ndarray,
        total_doc_ids: np.ndarray,
        offsets: List[int],
        lengths: List[int],
        n_docs: int
    ) -> List[List[Tuple[int, int]]]:
        """Find duplicate substring intervals to remove from documents.
        
        Args:
            suffix_array: Suffix array of concatenated string
            lcp_array: LCP array of concatenated string
            total_doc_ids: Document ID map
            offsets: Document starting positions
            lengths: Document lengths
            n_docs: Number of documents
            
        Returns:
            List of intervals to remove for each document
        """
        intervals_per_doc = [[] for _ in range(n_docs)]
        
        for i in range(len(lcp_array)):
            lcp_val = lcp_array[i]
            if lcp_val < self.threshold:
                continue
            
            pos1 = suffix_array[i]
            pos2 = suffix_array[i + 1]
            doc_id1 = total_doc_ids[pos1]
            doc_id2 = total_doc_ids[pos2]
            
            if doc_id1 < 0 or doc_id2 < 0 or doc_id1 == doc_id2:
                continue
            
            if doc_id1 < doc_id2:
                later_doc_id, pos_in_later = doc_id2, pos2
            elif doc_id2 < doc_id1:
                later_doc_id, pos_in_later = doc_id1, pos1
            else:
                continue
            
            start_in_doc = pos_in_later - offsets[later_doc_id]
            end_in_doc = start_in_doc + lcp_val - 1
            
            if not (0 <= start_in_doc < lengths[later_doc_id]):
                continue
            
            if end_in_doc >= lengths[later_doc_id]:
                end_in_doc = lengths[later_doc_id] - 1
                lcp_val = end_in_doc - start_in_doc + 1
                if lcp_val < self.threshold:
                    continue
            
            intervals_per_doc[later_doc_id].append((start_in_doc, end_in_doc))
        
        return intervals_per_doc
    
    def _merge_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping or adjacent intervals.
        
        Args:
            intervals: List of (start, end) intervals
            
        Returns:
            List of merged intervals
        """
        if not intervals:
            return []
        
        intervals.sort()
        merged = []
        cur_start, cur_end = intervals[0]
        
        for start, end in intervals[1:]:
            if start <= cur_end + 1:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        
        merged.append((cur_start, cur_end))
        return merged
    
    def _process_document_with_intervals(
        self,
        doc_bytes: bytes,
        intervals: List[Tuple[int, int]],
        original_doc: Document
    ) -> Document:
        """Process a document by removing specified intervals.
        
        Args:
            doc_bytes: Original document bytes
            intervals: Intervals to remove
            original_doc: Original Document object
            
        Returns:
            New Document object with duplicates removed
        """
        if not intervals:
            return original_doc
        
        merged_intervals = self._merge_intervals(intervals)
        new_bytes = bytearray()
        last_end = 0
        
        for start, end in merged_intervals:
            if last_end < start:
                new_bytes.extend(doc_bytes[last_end:start])
            last_end = end + 1
        
        if last_end < len(doc_bytes):
            new_bytes.extend(doc_bytes[last_end:])
        
        decode_func = tok_decode if self.enable_tokenizer else decode
        new_text = decode_func(bytes(new_bytes), self.tokenizer) if self.enable_tokenizer else decode_func(bytes(new_bytes))
        
        return replace(original_doc, text=new_text)

    def apply(self, chunk: Chunk) -> Optional[Chunk]:
        pass
    
    def apply_documents(self, documents: List[Document]) -> List[Document]:
        """Apply exact substring deduplication to documents.
        
        Args:
            documents: List of Document objects to process
            
        Returns:
            List of Document objects with duplicates removed
        """
        if not documents or len(documents) == 1:
            return documents.copy()
        
        doc_texts = [doc.text for doc in documents]
        doc_bytes_list = self._convert_documents_to_bytes(doc_texts)
        
        total_bytes, offsets, lengths = self._build_concatenated_string(doc_bytes_list)
        total_doc_ids = self._create_document_map(total_bytes, offsets, lengths)
        
        suffix_array = divsufsort(total_bytes)
        lcp_array = kasai(total_bytes, suffix_array)
        
        n_docs = len(documents)
        intervals_per_doc = self._find_duplicate_intervals(
            suffix_array, lcp_array, total_doc_ids, offsets, lengths, n_docs
        )
        
        result_documents = [documents[0]]
        
        for j in range(1, n_docs):
            result_documents.append(
                self._process_document_with_intervals(
                    doc_bytes_list[j],
                    intervals_per_doc[j],
                    documents[j]
                )
            )
        
        return result_documents


def tok_encode(seq, tokenizer):
    tokens = tokenizer.encode(seq)
    return struct.pack(f'<{len(tokens)}H', *tokens)


def tok_decode(byte_seq, tokenizer):
    num_tokens = len(byte_seq) // 2
    tokens = struct.unpack(f'<{num_tokens}H', byte_seq)
    return tokenizer.decode(list(tokens))


def encode(seq):
    return seq.encode('utf-8')


def decode(seq):
    return seq.decode('utf-8', errors='ignore')