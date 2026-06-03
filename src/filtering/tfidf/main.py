"""TfIdfInformativenessFilter — drop chunks with low TF-IDF density.

For each chunk we compute the mean TF-IDF weight per token (excluding
common stop-words).  Chunks whose density is below the configured
quantile are dropped.

Intuition:
    * Boilerplate / repeated headers / template phrases → low average
      TF-IDF (every word is common across the corpus).
    * Informative passages → higher TF-IDF (rarer, more specific words).

No LLM, no embedder, no external dependencies beyond scikit-learn —
purely lexical and universal across any text-corpus.
"""

import logging
import re
from typing import List

import numpy as np
from tqdm import tqdm

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"\b[A-Za-z_]\w*\b")


class TfIdfInformativenessFilter(Filter):
    """Drop chunks with low mean TF-IDF density.

    Args:
        low_quantile: chunks below this density quantile are dropped.
        min_chunk_tokens: chunks shorter than this are kept regardless
            (too few tokens to estimate density reliably).
        max_features: cap on vocabulary size for the TF-IDF matrix.
        ngram_range: (lo, hi) for word-n-grams.
        name: component name.
    """

    def __init__(
        self,
        low_quantile: float = 0.15,
        min_chunk_tokens: int = 5,
        max_features: int = 50000,
        ngram_range: tuple = (1, 1),
        name: str = "tfidf_filter",
    ) -> None:
        if not 0.0 <= low_quantile < 1.0:
            raise ValueError(f"Expect 0 <= low_quantile < 1, got {low_quantile}")
        self.low_quantile = float(low_quantile)
        self.min_chunk_tokens = int(min_chunk_tokens)
        self.max_features = int(max_features)
        self.ngram_range = tuple(ngram_range)
        self.name = name

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        n = len(chunks)
        if n == 0:
            return []

        texts = [chunk.text or "" for chunk in chunks]
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            token_pattern=_TOKEN_RE.pattern,
        )
        logger.info("TfIdfInformativenessFilter: fitting TF-IDF on %d chunks…", n)
        matrix = vectorizer.fit_transform(texts)

        # Mean TF-IDF per non-zero token in each chunk.
        densities = np.zeros(n, dtype=np.float64)
        token_counts = np.zeros(n, dtype=np.int64)
        for i in tqdm(range(n), desc="TfIdfInformativenessFilter density"):
            row = matrix.getrow(i)
            if row.nnz == 0:
                densities[i] = 0.0
                token_counts[i] = 0
            else:
                densities[i] = row.sum() / row.nnz
                token_counts[i] = row.nnz

        # Use only chunks with enough tokens for threshold estimation.
        eligible_mask = token_counts >= self.min_chunk_tokens
        if eligible_mask.sum() == 0:
            logger.warning(
                "TfIdfInformativenessFilter: no chunks have >= %d tokens; "
                "keeping all.",
                self.min_chunk_tokens,
            )
            return list(chunks)

        threshold = float(np.quantile(densities[eligible_mask], self.low_quantile))

        kept: List[Chunk] = []
        dropped = 0
        kept_short = 0
        for chunk, density, n_tokens in zip(chunks, densities, token_counts):
            if n_tokens < self.min_chunk_tokens:
                kept.append(chunk)
                kept_short += 1
                continue
            if density < threshold:
                dropped += 1
                continue
            kept.append(chunk)

        logger.info(
            "TfIdfInformativenessFilter: kept %d / %d "
            "(dropped %d below density quantile=%.2f, threshold=%.4f; "
            "kept %d short chunks unchanged)",
            len(kept), n, dropped, self.low_quantile, threshold, kept_short,
        )
        return kept
