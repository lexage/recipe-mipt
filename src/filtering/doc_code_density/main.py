"""DocumentCodeDensityFilter — pre-chunker version of CodeDensityFilter.

Drops whole Documents that contain very little code:
    * code_ratio = code_lines / total_non_empty_lines
    * (optionally) require at least one ast.parse-able block

Where CodeDensityFilter cleans chunks AFTER the chunker, this one cleans
whole documents BEFORE the chunker, sparing the embedder + Qdrant the
work of indexing prose-only pages.

Universal across any Python-code corpus — no library-name allow-list,
no benchmark-specific tags.
"""

import ast
import logging
import re
import textwrap
import warnings
from typing import List, Tuple

from tqdm import tqdm

from src.agent_constructor.core import Document
from src.agent_constructor.filters import DocumentFilter


logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(
    r"```(?:python|py|pycon)?\s*\n(?P<body>.*?)\n```",
    re.IGNORECASE | re.DOTALL,
)

_CODE_LINE_HEURISTIC = re.compile(
    r"""
    ^\s*(
        import\s | from\s.+\simport\s |
        def\s | class\s |
        return\s | return$ |
        if\s.+: | elif\s.+: | else\s*: |
        for\s.+: | while\s.+: |
        try\s*: | except | finally\s*: |
        with\s.+: |
        \w+\s*=\s*\S | \w+\s*\([^\)]* |
        @\w+ |
        \w+\.\w+ |
        \#.+ |
        \"{3} | \'{3} |
        raise\s | yield\s | pass$ | break$ | continue$
    )
    """,
    re.VERBOSE,
)


class DocumentCodeDensityFilter(DocumentFilter):
    """Drop documents whose code/prose ratio is below threshold.

    Args:
        min_code_ratio: minimum code/non-empty-lines ratio to keep a doc.
        require_valid_ast: also require at least one ast-parseable block.
        name: optional component name.
    """

    def __init__(
        self,
        min_code_ratio: float = 0.20,
        require_valid_ast: bool = True,
        name: str = "doc_code_density_filter",
    ) -> None:
        self.min_code_ratio = float(min_code_ratio)
        self.require_valid_ast = bool(require_valid_ast)
        self.name = name

    def apply(self, documents: List[Document]) -> List[Document]:
        kept: List[Document] = []
        dropped = 0
        for doc in tqdm(documents, desc="DocumentCodeDensityFilter"):
            text = doc.text or ""
            if not text.strip():
                dropped += 1
                continue
            if self._code_ratio(text) < self.min_code_ratio:
                dropped += 1
                continue
            if self.require_valid_ast and not self._has_valid_ast_block(text):
                dropped += 1
                continue
            kept.append(doc)

        logger.info(
            "DocumentCodeDensityFilter: kept %d / %d (dropped %d, "
            "min_ratio=%.2f, ast=%s)",
            len(kept), len(documents), dropped,
            self.min_code_ratio, self.require_valid_ast,
        )
        return kept

    @staticmethod
    def _split_code_and_prose(text: str) -> Tuple[List[str], List[str]]:
        code_blocks: List[str] = []
        leftover_parts: List[str] = []
        cursor = 0
        for m in _FENCE_RE.finditer(text):
            leftover_parts.append(text[cursor:m.start()])
            code_blocks.append(m.group("body"))
            cursor = m.end()
        leftover_parts.append(text[cursor:])
        return code_blocks, leftover_parts

    @classmethod
    def _code_ratio(cls, text: str) -> float:
        code_blocks, leftover_parts = cls._split_code_and_prose(text)
        code_lines = sum(
            sum(1 for ln in blk.splitlines() if ln.strip())
            for blk in code_blocks
        )
        leftover = "\n".join(leftover_parts)
        leftover_lines = [ln for ln in leftover.splitlines() if ln.strip()]
        for ln in leftover_lines:
            if _CODE_LINE_HEURISTIC.match(ln):
                code_lines += 1
        non_empty_total = sum(1 for ln in text.splitlines() if ln.strip())
        if non_empty_total == 0:
            return 0.0
        return code_lines / non_empty_total

    @classmethod
    def _has_valid_ast_block(cls, text: str) -> bool:
        code_blocks, leftover_parts = cls._split_code_and_prose(text)
        blocks_to_try = list(code_blocks)
        if not blocks_to_try:
            blocks_to_try = ["\n".join(leftover_parts)]
        for blk in blocks_to_try:
            candidate = textwrap.dedent(blk).strip()
            if not candidate:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    ast.parse(candidate)
                return True
            except (SyntaxError, ValueError):
                continue
        return False
