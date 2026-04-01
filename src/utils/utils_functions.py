import re

from typing import List

from src.agent_constructor.core import Chunk
from src.utils import DOCUMENT_SRC_DOCUMENTS, DOCUMENT_SRC_EXAMPLES

def replace_examples_in_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Заменяет плейсхолдеры <example_i> в чанках-документах на текст соответствующих примеров.
    
    Args:
        chunks: Список объектов Chunk
        
    Returns:
        Список чанков с замененными плейсхолдерами
    """
    examples_map = {}
    
    doc_chunks: List[Chunk] = []

    for chunk in chunks:
        if chunk.metadata.get("source") == DOCUMENT_SRC_EXAMPLES:
            doc_id = chunk.metadata.get("doc_id")
            order_id = chunk.metadata.get("order_id")
            
            if doc_id is not None and order_id is not None:
                key = (doc_id, order_id)
                if key in examples_map:
                    examples_map[key].append(chunk.text)
                else:
                    examples_map[key] = [chunk.text]
        else:
            doc_chunks.append(chunk)
    
    for key in examples_map:
        examples_map[key] = " ".join(examples_map[key])
    
    result_chunks = []
    
    for chunk in doc_chunks:
        
        if not (chunk.metadata.get("source") == DOCUMENT_SRC_DOCUMENTS):
            result_chunks.append(chunk)
            continue

        text = chunk.text
        doc_id = chunk.doc_id
        
        pattern = r'<example_(\d+)>'
        
        def replace_placeholder(match):
            order_id = int(match.group(1))
            key = (doc_id, order_id)
            
            if key in examples_map:
                return examples_map[key]

            return match.group(0)
        
        new_text = re.sub(pattern, replace_placeholder, text)
        
        if new_text != text:
            result_chunks.append(Chunk(
                id=chunk.id,
                doc_id=chunk.doc_id,
                text=new_text,
                tokens=chunk.tokens,
                metadata=chunk.metadata.copy()
            ))
        else:
            result_chunks.append(chunk)
    
    return result_chunks
