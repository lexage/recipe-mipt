from dataclasses import dataclass
from typing import List, Optional
from src.agent_constructor.core import Chunk, Document

@dataclass
class RagPath:

    query: str
    past_subqueries: Optional[List[str]]
    past_subanswers: Optional[List[str]]
    past_docs: Optional[List[List[Document]]]
    past_chunks: Optional[List[List[Chunk]]]

from dataclasses import dataclass
from typing import List, Optional
from src.agent_constructor.core import Chunk, Document

@dataclass
class RagPath:

    query: str
    past_subqueries: Optional[List[str]]
    past_subanswers: Optional[List[str]]
    past_docs: Optional[List[List[Document]]]
    past_chunks: Optional[List[List[Chunk]]]
