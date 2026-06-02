"""DocumentDedupFilter — Idea №7 from the experimental fileset.

Two-stage document-level deduplication, applied BEFORE chunking:

  1. Exact-hash pass:  normalise (lowercase + collapse whitespace), hash with
     MD5.  Documents with identical hashes form an exact-duplicate group.
  2. MinHash / LSH pass:  build MinHash signatures of word shingles on the
     survivors, query an LSH index with `near_dup_threshold` Jaccard.
     Documents in the same LSH bucket form a near-duplicate group.

Within each duplicate group we keep the longest document; the rest are
dropped.  This is safer than ExactSubstr (which cuts substrings inside
chunks) because we never mutate a survivor's text — we just drop full
duplicates before they ever reach the chunker.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List

from tqdm import tqdm

from src.agent_constructor.core import Document
from src.agent_constructor.filters import DocumentFilter


logger = logging.getLogger(__name__)


_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+")


class DocumentDedupFilter(DocumentFilter):
    """Drop duplicate / near-duplicate Documents before chunking.

    Args:
        near_dup_threshold: Jaccard threshold for the MinHash/LSH stage.
            Set to 1.0 to disable the near-dup stage (exact only).
        minhash_num_perm: number of permutations for MinHash. Higher → more
            accurate near-dup detection, but slower / more memory.
        shingle_size: word-n-gram length used for shingles.
        normalise_for_exact: if True, do `lowercase + collapse whitespace`
            before exact hashing.  Helps catch trivial-format duplicates.
        representative: 'longest' (default) or 'first' — which doc to keep
            in each duplicate group.
    """

    def __init__(
        self,
        near_dup_threshold: float = 0.85,
        minhash_num_perm: int = 128,
        shingle_size: int = 5,
        normalise_for_exact: bool = True,
        representative: str = "longest",
        name: str = "doc_dedup_filter",
    ) -> None:
        self.near_dup_threshold = float(near_dup_threshold)
        self.minhash_num_perm = int(minhash_num_perm)
        self.shingle_size = int(shingle_size)
        self.normalise_for_exact = bool(normalise_for_exact)
        if representative not in ("longest", "first"):
            raise ValueError("representative must be 'longest' or 'first'")
        self.representative = representative
        self.name = name

    # ------------------------------------------------------------------
    def apply(self, documents: List[Document]) -> List[Document]:
        n = len(documents)
        if n <= 1:
            return list(documents)

        # ---------- Stage 1: exact-hash ----------
        exact_groups: dict[str, list[int]] = {}
        for i, doc in enumerate(documents):
            h = self._exact_hash(doc.text or "")
            exact_groups.setdefault(h, []).append(i)

        kept_after_exact = self._pick_representatives(documents, exact_groups)
        n_after_exact = len(kept_after_exact)
        logger.info(
            "DocumentDedupFilter exact stage: %d → %d (-%d exact duplicates)",
            n, n_after_exact, n - n_after_exact,
        )

        # ---------- Stage 2: MinHash / LSH ----------
        if self.near_dup_threshold >= 1.0:
            return [documents[i] for i in sorted(kept_after_exact)]

        try:
            from datasketch import MinHash, MinHashLSH  # type: ignore
        except ImportError:
            logger.warning(
                "datasketch is not installed — skipping MinHash near-dup stage. "
                "pip install datasketch to enable it."
            )
            return [documents[i] for i in sorted(kept_after_exact)]

        lsh = MinHashLSH(threshold=self.near_dup_threshold,
                         num_perm=self.minhash_num_perm)
        signatures: dict[int, "MinHash"] = {}

        kept_idx_list = sorted(kept_after_exact)
        for i in tqdm(kept_idx_list, desc="DocumentDedupFilter MinHash"):
            mh = self._minhash(documents[i].text or "")
            signatures[i] = mh
            lsh.insert(str(i), mh)

        # Union-find on near-dup buckets.
        parent = {i: i for i in kept_idx_list}

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for i in kept_idx_list:
            neighbours = lsh.query(signatures[i])
            for sj in neighbours:
                j = int(sj)
                if j != i:
                    union(i, j)

        near_groups: dict[int, list[int]] = {}
        for i in kept_idx_list:
            near_groups.setdefault(find(i), []).append(i)

        kept_after_near = self._pick_representatives(documents, near_groups)
        n_after_near = len(kept_after_near)
        logger.info(
            "DocumentDedupFilter MinHash stage: %d → %d (-%d near duplicates, "
            "threshold=%.2f)",
            n_after_exact, n_after_near, n_after_exact - n_after_near,
            self.near_dup_threshold,
        )

        return [documents[i] for i in sorted(kept_after_near)]

    # ------------------------------------------------------------------
    def _exact_hash(self, text: str) -> str:
        if self.normalise_for_exact:
            normalised = _WS_RE.sub(" ", text.lower()).strip()
        else:
            normalised = text
        return hashlib.md5(normalised.encode("utf-8", errors="ignore")).hexdigest()

    def _minhash(self, text: str) -> "MinHash":  # type: ignore[name-defined]
        from datasketch import MinHash  # local import (already gated above)

        tokens = _TOKEN_RE.findall(text.lower())
        mh = MinHash(num_perm=self.minhash_num_perm)
        k = self.shingle_size
        if len(tokens) < k:
            # Fall back to single-token shingles for very short texts.
            for tok in tokens:
                mh.update(tok.encode("utf-8"))
            return mh
        for i in range(len(tokens) - k + 1):
            shingle = " ".join(tokens[i:i + k])
            mh.update(shingle.encode("utf-8"))
        return mh

    def _pick_representatives(
        self, documents: List[Document], groups: dict
    ) -> set[int]:
        keep: set[int] = set()
        for members in groups.values():
            if len(members) == 1:
                keep.add(members[0])
                continue
            if self.representative == "longest":
                rep = max(members, key=lambda k: len(documents[k].text or ""))
            else:
                rep = members[0]
            # Debug log to identify what got dedupped (helps catch over-merging)
            sample_lib = (
                documents[rep].metadata.get("library")
                if isinstance(documents[rep].metadata, dict) else None
            )
            logger.debug(
                "DocumentDedupFilter group: kept doc#%d (library=%s); dropped %d others.",
                rep, sample_lib, len(members) - 1,
            )
            keep.add(rep)
        return keep
