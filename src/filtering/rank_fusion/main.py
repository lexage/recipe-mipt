"""RankFusionFilter — parameter-light multi-signal quality filter.

Universal, domain-agnostic, no LLM.  Instead of one threshold per signal (the
2025 report's main critique of existing methods is hyperparameter
sensitivity), we compute several cheap text-quality signals, convert each to a
RANK, average the ranks, and drop the bottom `low_quantile` of the averaged
rank.  One knob.

Signals (all "higher = better", domain-agnostic):
  1. informativeness   — mean TF-IDF density (rare, specific words score high);
  2. lexical diversity — type/token ratio (repetitive boilerplate scores low);
  3. content ratio     — share of non-stopword word tokens.

Reworks tfidf + dedup-style redundancy into one robust composite — a NEW
filter, not any single named method.
"""

import logging
import re
from typing import List

import numpy as np

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[A-Za-z_]\w+")


class RankFusionFilter(Filter):
    """Drop the bottom `low_quantile` of chunks by averaged signal rank.

    Args:
        low_quantile: fraction of (eligible) chunks to drop by averaged rank.
        min_chunk_tokens: chunks shorter than this are kept as-is.
        max_features: vocabulary cap for the TF-IDF signal.
        name: component name.
    """

    def __init__(
        self,
        low_quantile: float = 0.15,
        min_chunk_tokens: int = 5,
        max_features: int = 50000,
        name: str = "rank_fusion_filter",
    ) -> None:
        if not 0.0 <= low_quantile < 1.0:
            raise ValueError(f"need 0 <= low_quantile < 1, got {low_quantile}")
        self.low_quantile = float(low_quantile)
        self.min_chunk_tokens = int(min_chunk_tokens)
        self.max_features = int(max_features)
        self.name = name

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        from sklearn.feature_extraction.text import (
            TfidfVectorizer,
            ENGLISH_STOP_WORDS,
        )

        n = len(chunks)
        if n <= 2:
            return list(chunks)

        texts = [c.text or "" for c in chunks]

        # Signal 1: TF-IDF mean density (vectorised).
        try:
            vec = TfidfVectorizer(
                max_features=self.max_features, stop_words="english"
            )
            m = vec.fit_transform(texts)
            sums = np.asarray(m.sum(axis=1)).ravel()
            nnz = np.diff(m.indptr)
            s1 = np.where(nnz > 0, sums / np.maximum(nnz, 1), 0.0)
        except ValueError:
            s1 = np.zeros(n)

        # Signals 2 & 3: lexical diversity and content ratio.
        s2 = np.zeros(n)
        s3 = np.zeros(n)
        ntok = np.zeros(n, dtype=np.int64)
        for i, t in enumerate(texts):
            toks = _TOKEN.findall(t.lower())
            ntok[i] = len(toks)
            if toks:
                s2[i] = len(set(toks)) / len(toks)
                content = sum(1 for w in toks if w not in ENGLISH_STOP_WORDS)
                s3[i] = content / len(toks)

        eligible_idx = np.where(ntok >= self.min_chunk_tokens)[0]
        if len(eligible_idx) < 3:
            return list(chunks)

        def rank01(values: np.ndarray) -> np.ndarray:
            order = np.argsort(np.argsort(values))
            return order / max(1, len(values) - 1)

        avg = (
            rank01(s1[eligible_idx])
            + rank01(s2[eligible_idx])
            + rank01(s3[eligible_idx])
        ) / 3.0

        thr = float(np.quantile(avg, self.low_quantile))
        drop_set = set(int(i) for i, a in zip(eligible_idx, avg) if a < thr)

        kept = [c for i, c in enumerate(chunks) if i not in drop_set]
        logger.info(
            "RankFusionFilter: kept %d / %d (dropped %d below avg-rank "
            "quantile=%.2f)",
            len(kept), n, len(drop_set), self.low_quantile,
        )
        return kept
