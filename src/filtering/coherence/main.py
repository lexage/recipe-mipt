"""CoherenceFilter — drop internally incoherent ("frankenstein") chunks.

Universal, domain-agnostic.  Badly assembled documents glue together unrelated
fragments (a scraping / merging artifact = noise).  We split each chunk into
sentences, embed them, and measure how tightly they cluster (mean cosine of
each sentence to the chunk centroid).  Low coherence = unrelated fragments →
drop.

Uses the project's embedder via dependency injection (same pattern as
SemanticDedupFilter).  A direct take on the grant's "robustness to noisy data"
requirement, and works on any domain.

NB: do NOT add `from __future__ import annotations` here — the registry injects
`embedder` by matching the `Agent` type, which breaks under string annotations.
"""

import logging
import re
from typing import List, Optional, Tuple

import numpy as np

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)

_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")


class CoherenceFilter(Filter):
    """Drop chunks whose sentences don't cohere around a single topic.

    Args:
        embedder: text→vector model, injected by the pipeline builder.
        min_coherence: keep chunks with mean sentence→centroid cosine >= this.
        min_sentences: chunks with fewer sentences are kept as-is.
        max_sentences: cap on sentences embedded per chunk (cost control).
        embed_batch_size: sentences per embedder call.
        name: component name.
    """

    def __init__(
        self,
        embedder: Agent,
        min_coherence: float = 0.5,
        min_sentences: int = 3,
        max_sentences: int = 12,
        embed_batch_size: int = 64,
        name: str = "coherence_filter",
    ) -> None:
        self.embedder = embedder
        self.min_coherence = float(min_coherence)
        self.min_sentences = int(min_sentences)
        self.max_sentences = int(max_sentences)
        self.embed_batch_size = int(embed_batch_size)
        self.name = name

    def _sentences(self, text: str) -> List[str]:
        parts = [p.strip() for p in _SENT.findall(text or "")]
        return [p for p in parts if len(p) >= 10]

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        n = len(chunks)
        if n == 0:
            return []

        # Collect (capped) sentences; flat list for one batched embed pass.
        per_chunk: List[Optional[Tuple[int, int]]] = []
        flat: List[str] = []
        for c in chunks:
            sents = self._sentences(c.text or "")[: self.max_sentences]
            if len(sents) < self.min_sentences:
                per_chunk.append(None)  # keep as-is (can't measure)
            else:
                per_chunk.append((len(flat), len(sents)))
                flat.extend(sents)

        if not flat:
            return list(chunks)

        emb = self._embed_all(flat)  # (T, d), L2-normalised

        kept: List[Chunk] = []
        dropped = 0
        for c, info in zip(chunks, per_chunk):
            if info is None:
                kept.append(c)
                continue
            start, k = info
            block = emb[start:start + k]
            centroid = block.mean(axis=0)
            cn = np.linalg.norm(centroid)
            if cn == 0:
                kept.append(c)
                continue
            coherence = float(np.mean(block @ (centroid / cn)))
            if coherence < self.min_coherence:
                dropped += 1
                continue
            kept.append(c)

        logger.info(
            "CoherenceFilter: kept %d / %d (dropped %d below coherence=%.2f)",
            len(kept), n, dropped, self.min_coherence,
        )
        return kept

    def _embed_all(self, texts: List[str]) -> np.ndarray:
        out: List[np.ndarray] = []
        bs = self.embed_batch_size
        for start in range(0, len(texts), bs):
            vectors = self.embedder.run(texts[start:start + bs])
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            out.append(arr)
        emb = np.concatenate(out, axis=0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms
