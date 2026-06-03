"""ASTComplexityFilter — iteration on CodeDensityFilter.

Score each parseable code block by AST *complexity* — number of nodes,
tree depth, and variety of node types — and drop chunks whose best
block is trivially simple (e.g. `import x`, single-line expressions).

A chunk with `import numpy as np` is technically valid Python, but
tells the model nothing about *how* to use numpy.  Chunks with control
flow / functions / classes / comprehensions are far more educational.

Universal: applies to any Python-code corpus.
"""

import ast
import logging
import re
import textwrap
import warnings
from typing import List, Tuple

from tqdm import tqdm

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(
    r"```(?:python|py|pycon)?\s*\n(?P<body>.*?)\n```",
    re.IGNORECASE | re.DOTALL,
)

INTERESTING_NODES = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith,
    ast.Try, ast.Raise, ast.Return, ast.Yield, ast.YieldFrom,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.Lambda, ast.Assert,
)


class ASTComplexityFilter(Filter):
    """Drop chunks whose Python AST is too simple to be educational."""

    def __init__(
        self,
        min_nodes: int = 15,
        min_depth: int = 3,
        min_interesting_nodes: int = 1,
        require_at_least_one_block: bool = True,
        name: str = "ast_complexity_filter",
    ) -> None:
        self.min_nodes = int(min_nodes)
        self.min_depth = int(min_depth)
        self.min_interesting_nodes = int(min_interesting_nodes)
        self.require_at_least_one_block = bool(require_at_least_one_block)
        self.name = name

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        kept: List[Chunk] = []
        dropped = 0
        for chunk in tqdm(chunks, desc="ASTComplexityFilter"):
            text = chunk.text or ""
            if not text.strip():
                dropped += 1
                continue
            scored = self._best_block_score(text)
            if scored is None:
                if self.require_at_least_one_block:
                    dropped += 1
                    continue
                kept.append(chunk)
                continue
            n_nodes, depth, n_interesting = scored
            if (
                n_nodes >= self.min_nodes
                and depth >= self.min_depth
                and n_interesting >= self.min_interesting_nodes
            ):
                kept.append(chunk)
            else:
                dropped += 1

        logger.info(
            "ASTComplexityFilter: kept %d / %d (dropped %d). "
            "Thresholds: nodes>=%d, depth>=%d, interesting>=%d",
            len(kept), len(chunks), dropped,
            self.min_nodes, self.min_depth, self.min_interesting_nodes,
        )
        return kept

    @staticmethod
    def _split_code_blocks(text: str) -> List[str]:
        blocks = [m.group("body") for m in _FENCE_RE.finditer(text)]
        if blocks:
            return blocks
        return [text]

    @classmethod
    def _best_block_score(cls, text: str):
        best = None
        for blk in cls._split_code_blocks(text):
            candidate = textwrap.dedent(blk).strip()
            if not candidate:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(candidate)
            except (SyntaxError, ValueError):
                continue
            score = cls._score_tree(tree)
            if best is None or score > best:
                best = score
        return best

    @staticmethod
    def _score_tree(tree: ast.AST) -> Tuple[int, int, int]:
        n_nodes = 0
        n_interesting = 0
        max_depth = 0

        def walk(node: ast.AST, depth: int) -> None:
            nonlocal n_nodes, n_interesting, max_depth
            n_nodes += 1
            if depth > max_depth:
                max_depth = depth
            if isinstance(node, INTERESTING_NODES):
                n_interesting += 1
            for child in ast.iter_child_nodes(node):
                walk(child, depth + 1)

        walk(tree, depth=0)
        return (n_nodes, max_depth, n_interesting)
