"""DocumentSemanticDedupFilter — document-level version of
SemanticDedupFilter.

Each Document is embedded as one vector (truncated to max_chars).
Documents whose cosine similarity exceeds the threshold are clustered;
one representative per cluster is kept.

Less aggressive than the chunk-level filter and much faster (thousands
of documents vs tens of thousands of chunks).
"""

import logging
from typing import List

import numpy as np
from tqdm import tqdm

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Document
from src.agent_constructor.filters import DocumentFilter


logger = logging.getLogger(__name__)


class DocumentSemanticDedupFilter(DocumentFilter):
    def __init__(
        self,
        embedder: Agent,
        similarity_threshold: float = 0.90,
        embed_batch_size: int = 16,
        compare_batch_size: int = 256,
        max_chars: int = 4000,
        representative: str = "longest",
        name: str = "doc_semantic_dedup_filter",
    ) -> None:
        self.embedder = embedder
        self.similarity_threshold = float(similarity_threshold)
        self.embed_batch_size = int(embed_batch_size)
        self.compare_batch_size = int(compare_batch_size)
        self.max_chars = int(max_chars)
        if representative not in ("longest", "first"):
            raise ValueError("representative must be 'longest' or 'first'")
        self.representative = representative
        self.name = name

    def apply(self, documents: List[Document]) -> List[Document]:
        n = len(documents)
        if n <= 1:
            return list(documents)

        texts = [(d.text or "")[: self.max_chars] for d in documents]
        embeddings = self._embed_all(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

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
        pbar = tqdm(range(0, n, bs), desc="DocumentSemanticDedup similarity")
        for start in pbar:
            end = min(start + bs, n)
            block = embeddings[start:end]
            sim_block = block @ embeddings.T
            for local_i, sim_row in enumerate(sim_block):
                i = start + local_i
                neighbours = np.where(sim_row[i + 1:] >= self.similarity_threshold)[0]
                for offset in neighbours:
                    union(i, i + 1 + int(offset))

        groups: dict = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        keep_indices = set()
        for members in groups.values():
            if len(members) == 1:
                keep_indices.add(members[0])
                continue
            if self.representative == "longest":
                rep = max(members, key=lambda k: len(documents[k].text or ""))
            else:
                rep = members[0]
            keep_indices.add(rep)

        kept = [documents[i] for i in sorted(keep_indices)]
        n_dup_groups = sum(1 for g in groups.values() if len(g) > 1)
        logger.info(
            "DocumentSemanticDedupFilter: kept %d / %d (groups=%d, "
            "duplicate groups=%d, threshold=%.3f)",
            len(kept), n, len(groups), n_dup_groups, self.similarity_threshold,
        )
        return kept

    def _embed_all(self, texts: List[str]) -> np.ndarray:
        out: list = []
        bs = self.embed_batch_size
        pbar = tqdm(range(0, len(texts), bs), desc="DocumentSemanticDedup embedding")
        for start in pbar:
            batch = texts[start:start + bs]
            vectors = self.embedder.run(batch)
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            out.append(arr)
        return np.concatenate(out, axis=0)
