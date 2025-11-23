from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Mapping,
    Optional,
)

# ---------- Basic types ----------

Text = str
Metadata = Mapping[str, Any]

@dataclass
class Document:
    id: str
    text: Text
    source: str  # e.g. 'github', 'stackoverflow', 'forum'
    metadata: Metadata = field(default_factory=dict)

@dataclass
class Chunk:
    id: str
    doc_id: str
    text: Text
    tokens: Optional[int] = None
    metadata: Metadata = field(default_factory=dict)

# ---------- Source & Stage enums ----------
class SourceKind(str, Enum):
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"
    FORUM = "forum"
    SNIPPET = "snippet"

class Stage(str, Enum):
    INGEST = "ingest"
    FILTER = "filter"
    AUGMENT = "augment"
    CONTEXT = "context"
    PIPELINE = "pipeline"


# ---------- Pluggable Blocks & Connectors ----------
class Block(ABC):
    """A generic block in the constructor — can be chunker/filter/augmenter/etc."""

    name: str

    def __init__(self, name: str):
        self.name = name


class Connector(ABC):
    """Defines compatibility between blocks (optional).

    In simple designs connectors are implicit — blocks accept/emit specific types.
    """

    @abstractmethod
    def is_compatible(self, a: Block, b: Block) -> bool:
        raise NotImplementedError
