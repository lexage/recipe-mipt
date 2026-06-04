"""ApiDocDensityFilter — keep chunks that actually document a callable API.

Code-specific (AST).  This is the ONE code-oriented filter in the portfolio
(the rest are domain-agnostic).  For a code corpus the useful chunks are API
reference entries: a signature (def/class), a Parameters/Returns section, maybe
an example.  Pure prose, navigation menus and fragments are dropped.

A chunk is kept when it has any of:
  * a parseable def/class (via ast on the whole text or a fenced block), OR
  * a signature-like line (`def f(`, `Class(`, `pandas.read_csv(`), OR
  * two or more docstring section markers (Parameters / Returns / Examples /
    numpy-style `----` underlines).

Our AST heuristic — no named published equivalent.
"""

import ast
import logging
import re
import textwrap
from typing import List

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:[\w+-]*)\n(.*?)```", re.DOTALL)
_SIG = re.compile(
    r"^\s*(?:def|class)\s+\w+|^\s*\w+(?:\.\w+)*\s*\(", re.MULTILINE
)
_SECTION = re.compile(
    r"(?im)^\s*(parameters|returns?|args|arguments|yields|raises|examples?|"
    r"attributes|notes|see also)\s*:?\s*$|^\s*-{3,}\s*$"
)


class ApiDocDensityFilter(Filter):
    """Keep chunks that document a callable; drop prose / nav / fragments.

    Args:
        require_signature: if True, a chunk needs a signature OR >=2 section
            markers; if False, a single section marker is enough.
        min_section_markers: section markers required in the looser mode.
        name: component name.
    """

    def __init__(
        self,
        require_signature: bool = True,
        min_section_markers: int = 1,
        name: str = "api_doc_filter",
    ) -> None:
        self.require_signature = bool(require_signature)
        self.min_section_markers = int(min_section_markers)
        self.name = name

    @staticmethod
    def _has_callable_ast(text: str) -> bool:
        candidates = _FENCE.findall(text) or [text]
        for block in candidates:
            try:
                tree = ast.parse(textwrap.dedent(block))
            except (SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    return True
        return False

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        n = len(chunks)
        if n == 0:
            return []

        kept: List[Chunk] = []
        dropped = 0
        for c in chunks:
            text = c.text or ""
            has_sig = bool(_SIG.search(text)) or self._has_callable_ast(text)
            n_sections = len(_SECTION.findall(text))

            if self.require_signature:
                keep = has_sig or n_sections >= 2
            else:
                keep = has_sig or n_sections >= self.min_section_markers

            if keep:
                kept.append(c)
            else:
                dropped += 1

        logger.info(
            "ApiDocDensityFilter: kept %d / %d (dropped %d non-API chunks)",
            len(kept), n, dropped,
        )
        return kept
