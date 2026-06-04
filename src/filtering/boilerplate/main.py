"""BoilerplateFilter — drop template chunks by cross-corpus line frequency.

Universal, domain-agnostic, no LLM.  Boilerplate (license headers, navigation
menus, repeated disclaimers, empty "Parameters / Returns" skeletons) appears
almost identically across MANY documents.  We compute, over the whole corpus,
how often each normalised line occurs (document frequency), then drop chunks
that consist mostly of such high-frequency template lines.

Reworks the IDF idea toward NOISE REMOVAL (rather than informativeness
scoring).  Works the same on legal disclaimers, medical warnings or code
license headers.
"""

import logging
import re
from collections import Counter
from typing import List

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


class BoilerplateFilter(Filter):
    """Drop chunks dominated by lines that repeat across the corpus.

    Args:
        max_doc_frequency: a normalised line occurring in >= this fraction of
            chunks is treated as a template line.
        drop_ratio: drop a chunk if this fraction of its lines are template.
        min_lines: chunks with fewer lines are kept as-is.
        min_line_chars: ignore lines shorter than this when counting.
        name: component name.
    """

    def __init__(
        self,
        max_doc_frequency: float = 0.5,
        drop_ratio: float = 0.6,
        min_lines: int = 3,
        min_line_chars: int = 4,
        name: str = "boilerplate_filter",
    ) -> None:
        self.max_doc_frequency = float(max_doc_frequency)
        self.drop_ratio = float(drop_ratio)
        self.min_lines = int(min_lines)
        self.min_line_chars = int(min_line_chars)
        self.name = name

    def _lines(self, text: str) -> List[str]:
        out = []
        for raw in (text or "").splitlines():
            ln = _WS.sub(" ", raw.strip().lower())
            if len(ln) >= self.min_line_chars:
                out.append(ln)
        return out

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        n = len(chunks)
        if n == 0:
            return []

        # Document frequency of each normalised line (count once per chunk).
        df: Counter = Counter()
        chunk_lines: List[List[str]] = []
        for c in chunks:
            lines = self._lines(c.text or "")
            chunk_lines.append(lines)
            for ln in set(lines):
                df[ln] += 1

        threshold_count = max(2, int(self.max_doc_frequency * n))

        kept: List[Chunk] = []
        dropped = 0
        for c, lines in zip(chunks, chunk_lines):
            if len(lines) < self.min_lines:
                kept.append(c)
                continue
            common = sum(1 for ln in lines if df[ln] >= threshold_count)
            share = common / len(lines)
            if share >= self.drop_ratio:
                dropped += 1
                continue
            kept.append(c)

        logger.info(
            "BoilerplateFilter: kept %d / %d (dropped %d; template line if seen "
            "in >= %d chunks)",
            len(kept), n, dropped, threshold_count,
        )
        return kept
