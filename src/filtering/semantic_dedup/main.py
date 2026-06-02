"""SemanticDedupFilter — Idea №2 from the experimental fileset.

A safer replacement for ExactSubstrFiltrator.  Where ExactSubstr cuts
byte-level substrings *inside* chunks (and breaks them), this filter
drops near-duplicate chunks as *whole units*:

  1. Embed each chunk via the same embedder the Qdrant index uses
     (Qwen3-Embedding-4B in our setup).
  2. Compute the cosine-similarity matrix in batches (memory-friendly).
  3. Build a graph: chunks with similarity >= threshold are connected.
  4. Keep one representative per connected component.

Architecturally a corpus-level filter — same API as LengthFilter, called
once during pipeline build.  Uses the project's embedder via dependency
injection (the registry recognises type `Agent` as an injectable Block).
"""

import logging
from typing import List

import numpy as np
from tqdm import tqdm

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)


class SemanticDedupFilter(Filter):
    """Drop near-duplicate chunks based on embedding cosine similarity.

    Args:
        embedder: text→vector model, injected by the pipeline builder.
        similarity_threshold: cosine sim >= this counts as a duplicate.
        embed_batch_size: chunks per embedder call.
        compare_batch_size: rows per matmul block when finding neighbours.
        representative: 'longest' keeps the chunk with the longest text in
            each duplicate group; 'first' keeps the first one encountered.
    """

    def __init__(
        self,
        embedder: Agent,
        similarity_threshold: float = 0.92,
        embed_batch_size: int = 32,
        compare_batch_size: int = 1024,
        representative: str = "longest",
        name: str = "semantic_dedup_filter",
    ) -> None:
        self.embedder = embedder
        self.similarity_threshold = float(similarity_threshold)
        self.embed_batch_size = int(embed_batch_size)
        self.compare_batch_size = int(compare_batch_size)
        if representative not in ("longest", "first"):
            raise ValueError("representative must be 'longest' or 'first'")
        self.representative = representative
        self.name = name

    # ------------------------------------------------------------------
    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        n = len(chunks)
        if n <= 1:
            return list(chunks)

        # 1. Embed all chunks ------------------------------------------------
        texts = [c.text or "" for c in chunks]
        embeddings = self._embed_all(texts)
        # Normalise rows → cosine sim becomes plain dot product.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        # 2 & 3. Build duplicate graph via batched matmul -------------------
        parent = list(range(n))

        def find(i: int) -> int:
            root = i
            while parent[root] != root:
                root = parent[root]
            while parent[i] != root:
                parent[i], i = root, parent[i]
            return root

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        bs = self.compare_batch_size
        pbar = tqdm(range(0, n, bs), desc="SemanticDedup similarity")
        for start in pbar:
            end = min(start + bs, n)
            block = embeddings[start:end]  # (b, d)
            sim_block = block @ embeddings.T  # (b, n)
            # Mask self-similarity and lower triangle so each pair is
            # considered once.
            for local_i, sim_row in enumerate(sim_block):
                i = start + local_i
                # Only inspect j > i; for j <= i it's already handled.
                neighbours = np.where(sim_row[i + 1:] >= self.similarity_threshold)[0]
                for offset in neighbours:
                    j = i + 1 + int(offset)
                    union(i, j)

        # 4. Group + pick representative ------------------------------------
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        keep_indices = set()
        for members in groups.values():
            if len(members) == 1:
                keep_indices.add(members[0])
                continue
            if self.representative == "longest":
                rep = max(members, key=lambda k: len(chunks[k].text or ""))
            else:
                rep = members[0]
            keep_indices.add(rep)

        kept = [chunks[i] for i in sorted(keep_indices)]

        n_groups = len(groups)
        n_dup_groups = sum(1 for g in groups.values() if len(g) > 1)
        logger.info(
            "SemanticDedupFilter: kept %d / %d (groups=%d, duplicate groups=%d, threshold=%.3f)",
            len(kept), n, n_groups, n_dup_groups, self.similarity_threshold,
        )
        return kept

    # ------------------------------------------------------------------
    def _embed_all(self, texts: List[str]) -> np.ndarray:
        """Embed `texts` in batches.  Returns float32 array (n, d)."""
        out: list[np.ndarray] = []
        bs = self.embed_batch_size
        pbar = tqdm(range(0, len(texts), bs), desc="SemanticDedup embedding")
        for start in pbar:
            batch = texts[start:start + bs]
            vectors = self.embedder.run(batch)
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            out.append(arr)
        return np.concatenate(out, axis=0)
