"""CompressionDensityFilter — information density via compressibility.

Universal, domain-agnostic, no LLM.  Idea: boilerplate / repetitive text
compresses very well (small compressed/original ratio → little information);
informative prose compresses less; random / garbled / encoding-noise text is
nearly incompressible (very high ratio).  We keep the "interesting middle" of
the compression-ratio distribution and drop both tails.

This reworks the perplexity-filter intuition (keep the middle of the
distribution) but WITHOUT a language model — a cheap proxy that works on any
language or domain.  Directly answers the 2025 report's main critique of
existing methods (compute cost / dependence on big models).
"""

import logging
import zlib
from typing import List

import numpy as np

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)


class CompressionDensityFilter(Filter):
    """Drop chunks at the tails of the compression-ratio distribution.

    Args:
        low_quantile: drop chunks below this ratio quantile (boilerplate).
        high_quantile: drop chunks above this ratio quantile (garbage/noise).
        min_chars: chunks shorter than this are kept as-is (ratio unreliable).
        name: component name.
    """

    def __init__(
        self,
        low_quantile: float = 0.10,
        high_quantile: float = 0.98,
        min_chars: int = 80,
        name: str = "compression_filter",
    ) -> None:
        if not 0.0 <= low_quantile < high_quantile <= 1.0:
            raise ValueError(
                f"need 0 <= low < high <= 1, got {low_quantile}, {high_quantile}"
            )
        self.low_quantile = float(low_quantile)
        self.high_quantile = float(high_quantile)
        self.min_chars = int(min_chars)
        self.name = name

    @staticmethod
    def _ratio(text: str) -> float:
        raw = (text or "").encode("utf-8", errors="ignore")
        if len(raw) == 0:
            return 0.0
        return len(zlib.compress(raw, 6)) / len(raw)

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        n = len(chunks)
        if n == 0:
            return []

        ratios = np.array([self._ratio(c.text or "") for c in chunks])
        lengths = np.array([len(c.text or "") for c in chunks])
        eligible = lengths >= self.min_chars
        if eligible.sum() < 2:
            return list(chunks)

        lo = float(np.quantile(ratios[eligible], self.low_quantile))
        hi = float(np.quantile(ratios[eligible], self.high_quantile))

        kept: List[Chunk] = []
        dropped = 0
        for c, r, length in zip(chunks, ratios, lengths):
            if length < self.min_chars:
                kept.append(c)
                continue
            if r < lo or r > hi:
                dropped += 1
                continue
            kept.append(c)

        logger.info(
            "CompressionDensityFilter: kept %d / %d (dropped %d outside "
            "ratio [%.3f, %.3f])",
            len(kept), n, dropped, lo, hi,
        )
        return kept
