"""CodeDensityFilter — Idea №1 from the experimental fileset.

For each chunk:
  1. Extract code blocks (markdown ``` fences and inline indented blocks).
  2. Attempt ast.parse() on each block (with textwrap.dedent + tolerance).
  3. Compute code_ratio = code_lines / total_non_empty_lines.

A chunk is kept when:
  * code_ratio >= min_code_ratio, AND
  * (if require_valid_ast) at least one block parses cleanly.

The filter is corpus-level, has no query awareness, no learned parameters,
and does not call any external service. Cheap.
"""

import ast
import logging
import re
import textwrap
from typing import List, Tuple

from tqdm import tqdm

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)

# Markdown fenced code blocks. Optionally with a language tag.
_FENCE_RE = re.compile(
    r"```(?:python|py|pycon)?\s*\n(?P<body>.*?)\n```",
    re.IGNORECASE | re.DOTALL,
)

# Lines that look like Python code outside of any fence.
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


class CodeDensityFilter(Filter):
    """Keep chunks with enough Python code AND syntactically valid blocks.

    Args:
        min_code_ratio: minimum (code lines / non-empty lines) ratio to keep.
        require_valid_ast: if True, also require at least one ast-parseable
            block. If a chunk has no fenced blocks, the whole chunk is parsed
            as a single block.
        name: optional component name (unused in logic).
    """

    def __init__(
        self,
        min_code_ratio: float = 0.25,
        require_valid_ast: bool = True,
        name: str = "code_density_filter",
    ) -> None:
        self.min_code_ratio = float(min_code_ratio)
        self.require_valid_ast = bool(require_valid_ast)
        self.name = name

    # ------------------------------------------------------------------
    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        kept: List[Chunk] = []
        dropped = 0
        for chunk in tqdm(chunks, desc="CodeDensityFilter"):
            text = chunk.text or ""
            if not text.strip():
                dropped += 1
                continue

            code_ratio = self._code_ratio(text)
            if code_ratio < self.min_code_ratio:
                dropped += 1
                continue

            if self.require_valid_ast and not self._has_valid_ast_block(text):
                dropped += 1
                continue

            kept.append(chunk)

        logger.info(
            "CodeDensityFilter: kept %d / %d (dropped %d, min_ratio=%.2f, ast=%s)",
            len(kept), len(chunks), dropped,
            self.min_code_ratio, self.require_valid_ast,
        )
        return kept

    # ------------------------------------------------------------------
    @staticmethod
    def _split_code_and_prose(text: str) -> Tuple[List[str], List[str]]:
        """Return (code_blocks, remaining_text_without_fences)."""
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
            # No fences — try the whole chunk as one block.
            blocks_to_try = ["\n".join(leftover_parts)]

        for blk in blocks_to_try:
            candidate = textwrap.dedent(blk).strip()
            if not candidate:
                continue
            try:
                ast.parse(candidate)
                return True
            except (SyntaxError, ValueError):
                continue
        return False
