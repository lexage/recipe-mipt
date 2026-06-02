"""AggressivePatternFilter — Idea №6 from the experimental fileset.

A stricter junk filter than SimpleLexicalFiltrator: drops whole chunks
that match any of these aggressive patterns:

  * ToC-like (numbered sections only, no body)
  * URL-heavy (>50% of characters belong to URLs / markdown links)
  * Non-ASCII heavy (>30% non-ASCII — likely OCR / encoding noise)
  * Too short after stripping punctuation (<100 chars of real content)
  * Structure-headers-only (only "Parameters", "Returns", "Examples", …
    without any meaningful body underneath)

Drops the entire chunk rather than editing it.  Safer than
SimpleLexicalFiltrator's in-place mutation.
"""

import logging
import re
import string
from typing import List

from tqdm import tqdm

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)


_URL_RE = re.compile(
    r"https?://\S+|"                # bare URLs
    r"\[[^\]]+\]\([^)]+\)|"         # markdown [text](url)
    r"<[^>]+@[^>]+>"                # raw <email>
)

_TOC_NUMBER_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s+\S")
_TOC_CHAPTER_RE = re.compile(r"^\s*(Chapter|Section|Part)\s+\d", re.IGNORECASE)

_STRUCTURE_HEADERS = {
    "parameters", "returns", "examples", "example", "see also",
    "notes", "note", "references", "raises", "yields", "warnings",
    "arguments", "args", "kwargs", "attributes", "methods",
}


class AggressivePatternFilter(Filter):
    """Drop chunks matching one of the configured 'junk' shapes."""

    def __init__(
        self,
        max_url_ratio: float = 0.5,
        max_non_ascii_ratio: float = 0.3,
        min_content_chars: int = 100,
        drop_toc: bool = True,
        drop_structure_only: bool = True,
        name: str = "aggressive_pattern_filter",
    ) -> None:
        self.max_url_ratio = float(max_url_ratio)
        self.max_non_ascii_ratio = float(max_non_ascii_ratio)
        self.min_content_chars = int(min_content_chars)
        self.drop_toc = bool(drop_toc)
        self.drop_structure_only = bool(drop_structure_only)
        self.name = name

    # ------------------------------------------------------------------
    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        kept: List[Chunk] = []
        drop_reasons = {
            "empty": 0, "url_heavy": 0, "non_ascii_heavy": 0,
            "too_short": 0, "toc": 0, "structure_only": 0,
        }
        for chunk in tqdm(chunks, desc="AggressivePatternFilter"):
            text = chunk.text or ""
            reason = self._junk_reason(text)
            if reason is None:
                kept.append(chunk)
            else:
                drop_reasons[reason] += 1

        logger.info(
            "AggressivePatternFilter: kept %d / %d. Drop reasons: %s",
            len(kept), len(chunks), drop_reasons,
        )
        return kept

    # ------------------------------------------------------------------
    def _junk_reason(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return "empty"

        # URL density
        url_chars = sum(len(m.group(0)) for m in _URL_RE.finditer(text))
        if len(text) and url_chars / len(text) > self.max_url_ratio:
            return "url_heavy"

        # Non-ASCII density (after removing whitespace).
        chars_no_ws = "".join(text.split())
        if chars_no_ws:
            non_ascii = sum(1 for c in chars_no_ws if ord(c) > 127)
            if non_ascii / len(chars_no_ws) > self.max_non_ascii_ratio:
                return "non_ascii_heavy"

        # Content-length after stripping punctuation and whitespace.
        content_only = stripped.translate(
            str.maketrans("", "", string.punctuation + string.whitespace)
        )
        if len(content_only) < self.min_content_chars:
            return "too_short"

        lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]

        # ToC-like: almost all lines look like numbered entries.
        if self.drop_toc and len(lines) >= 3:
            toc_like = sum(
                1 for ln in lines
                if _TOC_NUMBER_RE.match(ln) or _TOC_CHAPTER_RE.match(ln)
            )
            if toc_like / len(lines) > 0.7:
                return "toc"

        # Structure-only: lines are all headers like "Parameters" / "Returns"
        # without code or substantive text in between.
        if self.drop_structure_only:
            non_header_chars = 0
            for ln in lines:
                clean = ln.rstrip(":").strip().lower()
                if clean in _STRUCTURE_HEADERS:
                    continue
                non_header_chars += len(ln)
            if non_header_chars < self.min_content_chars:
                return "structure_only"

        return None
