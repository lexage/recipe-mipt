"""ApiReferenceSectionFilter — keep only API-reference documents by SECTION label.

Faithful reproducer of the manual ``docs_database_examples_apis.db`` filtration.
Empirically (see CLAUDE.md, exp6 analysis) the manual DB was produced by exactly
one rule: keep every document whose ``section`` is ``reference`` and drop every
document in ``user_guide`` / ``tutorial`` / ``examples`` / ``dev_notes`` — a pure
document-level deletion, retained docs byte-identical, examples follow their doc.

This filter is the *oracle / ceiling*: it relies on the corpus carrying a clean
section taxonomy (``Document.metadata['section']``, populated by the sqlite
adapter from ``results/<lib>_<section>``), so it reproduces ``apis.db`` exactly
here but is NOT universal — see ``ApiReferenceGenreFilter`` for the content-based
counterpart that needs no metadata.

Document-level filter: used by ``SimplePipelineWithDocFilter`` (pipeline type
``SIMPLE_WITH_DOC_FILTER``), runs BEFORE chunking.
"""

from typing import List

from src.agent_constructor.core import Document
from src.agent_constructor.filters import DocumentFilter


class ApiReferenceSectionFilter(DocumentFilter):
    """Keep documents whose ``metadata['section']`` is in ``keep_sections``.

    Args:
        name: block name.
        keep_sections: section names to keep (default: ``["reference"]``).
            Everything else is dropped.
    """

    def __init__(
        self,
        name: str = "api_section_filter",
        keep_sections: List[str] = None,
    ):
        super().__init__(name)
        self.keep_sections = set(keep_sections) if keep_sections else {"reference"}

    def apply(self, documents: List[Document]) -> List[Document]:
        kept = [
            doc
            for doc in documents
            if (doc.metadata or {}).get("section") in self.keep_sections
        ]
        return kept
