"""SelfConsistencyCleaner — clean text using the text ITSELF as the reference.

Our NEW mechanism (научная новизна): every known filter compares text to
something EXTERNAL — corpus frequencies (dedup / IDF), a trained classifier,
a target distribution (DSIR), big-model perplexity, or an LLM judge (see the
2025 report, §3.1). We instead ask, at every granularity: *does this piece fit
the piece that contains it?* Injected dirt is a foreign body that falls out of
its host. We keep the self-consistent core and CUT what does not belong; a
chunk is dropped only if nothing survives.

This is a CLEANER (not a selector), at THREE granularities:

  * LEVEL 1 — a word vs itself (cheap, NON-semantic). Junk symbols inserted
    INTO a word (e.g. "ar#r@ay") are stripped when glued to letters, so it
    becomes "array"; code (`np.array(`), numbers and domain terms (`int64`,
    `float32`, `read_csv`) are left intact. Letter/digit corruption without
    symbols ("arr4ay", "arxray") is left for level 3.

  * LEVEL 2 — a line vs the LINES THAT BELONG TOGETHER (semantic). We do NOT
    take a document mean (with most lines dirty the mean is dirt). Each line's
    neighbour-density (agreement with its closest DISTINCT lines) tells whether
    it is part of a meaningful group or an outlier. The cut is relative —
    derived from the document's own density distribution — and VERBATIM
    duplicates are discounted (by exact text) so repeated junk can't vouch for
    its own copies while genuinely similar good lines still vouch.

  * LEVEL 3 — a word vs its SENTENCE, variant C (hybrid). Inside the lines kept
    by level 2 we drop alien words with two cooperating signals:
      (A) character normality, no embedder: a character n-gram model learnt
          from the corpus's OWN words scores how "word-like" each token is.
          Foreign-script words, random gibberish ("qwzlkj") and letter-
          corrupted tokens ("arxray") score far below the corpus and are cut.
      (B) semantics, for the rest: for a still-suspect word we embed the line
          WITH and WITHOUT it and keep it only if removing it does NOT pull the
          line closer to the level-2 CLEANED core (full-sentence embeddings —
          robust — and the reference is the already-cleaned good lines, not the
          noisy raw document). This catches real-but-off-topic words.
    Only non-recurring / foreign words are even considered, so recurring domain
    terms are never touched.

NB: do NOT add `from __future__ import annotations` here — the registry injects
`embedder` by matching the `Agent` type, which breaks under string annotations.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_LETTER_RUN = re.compile(r"[A-Za-z]+")
_JUNK = "#$%^&*~`!@"
_JC = re.escape(_JUNK)
_L1_RE = re.compile(rf"(?<=[A-Za-z])[{_JC}]|[{_JC}](?=[A-Za-z])")


class SelfConsistencyCleaner(Filter):
    """Clean dirt that does not fit its own word / line / sentence.

    Args:
        embedder: text->vector model, injected by the builder (the `embedder`
            component). Used for levels 2 and 3-B.
        min_line_chars: lines shorter than this are kept as-is.
        garbled_alpha_ratio, garbled_max_run: a line that is mostly non-letters
            with no real word-run is symbol soup -> dropped (cheap pre-clean).
        knn: neighbours used for the level-2 density estimate.
        sep_z: level-2 relative margin — a line is cut only if its density sits
            more than `sep_z` core-std-devs below the coherent group.
        min_doc_lines: documents with fewer substantial lines skip level 2
            (and therefore level 3-B; level 3-A still runs).
        clean_words: enable level 3 (per-word cleaning, variant C).
        word_min_len: minimum letter-length of a "content word".
        min_sentence_words: a line needs at least this many content words to run
            level 3.
        char_z: level 3-A margin — a candidate word is cut if its character
            n-gram normality sits more than `char_z` MADs below the corpus.
        semantic_word_check: enable level 3-B (semantic leave-one-out).
        word_improve: level 3-B margin — remove a word only if dropping it pulls
            the line at least this much closer to the cleaned core (cosine).
        max_word_cut_frac: never remove more than this fraction of a line's words.
        embed_batch_size: texts per embedder call.
        min_keep_chars: drop a chunk whose cleaned text is shorter than this.
        name: component name.
    """

    def __init__(
        self,
        embedder: Agent,
        min_line_chars: int = 12,
        garbled_alpha_ratio: float = 0.6,
        garbled_max_run: int = 3,
        knn: int = 4,
        sep_z: float = 2.0,
        min_doc_lines: int = 5,
        clean_words: bool = True,
        word_min_len: int = 3,
        min_sentence_words: int = 4,
        char_z: float = 2.5,
        semantic_word_check: bool = True,
        word_improve: float = 0.05,
        max_word_cut_frac: float = 0.5,
        embed_batch_size: int = 64,
        min_keep_chars: int = 1,
        name: str = "self_clean_filter",
    ) -> None:
        self.embedder = embedder
        self.min_line_chars = int(min_line_chars)
        self.garbled_alpha_ratio = float(garbled_alpha_ratio)
        self.garbled_max_run = int(garbled_max_run)
        self.knn = int(knn)
        self.sep_z = float(sep_z)
        self.min_doc_lines = int(min_doc_lines)
        self.clean_words = bool(clean_words)
        self.word_min_len = int(word_min_len)
        self.min_sentence_words = int(min_sentence_words)
        self.char_z = float(char_z)
        self.semantic_word_check = bool(semantic_word_check)
        self.word_improve = float(word_improve)
        self.max_word_cut_frac = float(max_word_cut_frac)
        self.embed_batch_size = int(embed_batch_size)
        self.min_keep_chars = int(min_keep_chars)
        self.name = name

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _norm(line: str) -> str:
        return _WS.sub(" ", line.strip().lower())

    def _l1_repair(self, line: str) -> str:
        """Level 1: strip junk symbols glued to letters inside words."""
        return _L1_RE.sub("", line)

    def _is_garbled(self, line: str) -> bool:
        s = line.strip()
        compact = s.replace(" ", "")
        if not compact:
            return False
        letters = sum(c.isalpha() for c in compact)
        alpha_ratio = letters / len(compact)
        runs = _LETTER_RUN.findall(s)
        max_run = max((len(r) for r in runs), default=0)
        return alpha_ratio < self.garbled_alpha_ratio and max_run <= self.garbled_max_run

    @staticmethod
    def _word_core(tok: str) -> str:
        return "".join(c for c in tok if c.isalpha())

    def _embed(self, texts: List[str]) -> np.ndarray:
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

    def _keep_mask(self, density: np.ndarray) -> np.ndarray:
        """Level-2 relative cut from the document's own density distribution."""
        m = len(density)
        if m < 3:
            return np.ones(m, dtype=bool)
        order = np.argsort(density)
        ds = density[order]
        csum = np.cumsum(ds)
        total = float(csum[-1])
        best_t, best_sep = 1, -1.0
        for t in range(1, m):
            s1 = float(csum[t - 1])
            mu1 = s1 / t
            mu2 = (total - s1) / (m - t)
            sep = t * (m - t) * (mu2 - mu1) ** 2
            if sep > best_sep:
                best_sep, best_t = sep, t
        low_idx = order[:best_t]
        high_idx = order[best_t:]
        mu_high = float(density[high_idx].mean())
        std_high = float(density[high_idx].std())
        mu_low = float(density[low_idx].mean())
        thr = mu_high - self.sep_z * std_high
        keep = np.ones(m, dtype=bool)
        if mu_low < thr:
            keep = density >= thr
        return keep

    # ---- level 3-A: character n-gram normality (no embedder) -----------

    @staticmethod
    def _build_char_model(words: List[str]):
        tg: Counter = Counter()
        total = 0
        for w in words:
            s = "^" + w + "$"
            for i in range(len(s) - 2):
                tg[s[i:i + 3]] += 1
                total += 1
        return tg, total, max(1, len(tg))

    def _char_score(self, w: str, model) -> float:
        """Mean of the two least-likely char trigrams (lower = less word-like)."""
        tg, total, vocab = model
        s = "^" + w + "$"
        if len(s) < 3:
            return 0.0
        lps = []
        denom = total + vocab
        for i in range(len(s) - 2):
            c = tg.get(s[i:i + 3], 0)
            lps.append(math.log((c + 1) / denom))
        lps.sort()
        k = min(2, len(lps))
        return sum(lps[:k]) / k

    # ---- level 3: clean alien words inside a kept line -----------------

    def _clean_sentence(self, line, vocab, model, char_thr, core, line_emb) -> str:
        toks = line.split()
        content = []
        for j, tok in enumerate(toks):
            core_w = self._word_core(tok)
            if len(core_w) >= self.word_min_len:
                content.append((j, core_w.lower(), any(ord(c) > 127 for c in core_w)))
        if len(content) < self.min_sentence_words:
            return line

        # only non-recurring / foreign words are candidates
        cand = [c for c in content if c[2] or vocab[c[1]] <= 1]
        if not cand:
            return line

        remove = set()
        semantic_cand = []
        for j, low, foreign in cand:
            if self._char_score(low, model) < char_thr:    # (A) not word-like
                remove.add(j)
            else:
                semantic_cand.append((j, low))

        # (B) semantics: keep a still-suspect word unless dropping it pulls the
        # line closer to the level-2 cleaned core.
        if (self.semantic_word_check and core is not None
                and line_emb is not None and semantic_cand):
            base_sim = float(line_emb @ core)
            variants = [
                " ".join(t for k, t in enumerate(toks) if k != j)
                for j, _ in semantic_cand
            ]
            ve = self._embed(variants)
            for (j, _low), v in zip(semantic_cand, ve):
                if float(v @ core) - base_sim > self.word_improve:
                    remove.add(j)

        if not remove:
            return line
        max_cut = max(1, int(self.max_word_cut_frac * len(content)))
        if len(remove) > max_cut:
            remove = set(sorted(remove)[:max_cut])
        return " ".join(t for k, t in enumerate(toks) if k not in remove)

    # ---- main -----------------------------------------------------------

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        n = len(chunks)
        if n == 0:
            return []

        # Level 1 on every line first (cheap, word-local).
        repaired: List[List[str]] = [
            [self._l1_repair(ln) for ln in (c.text or "").split("\n")]
            for c in chunks
        ]

        # Global character model + relative threshold (level 3-A), built once.
        char_model = char_thr = None
        if self.clean_words:
            corpus_words: List[str] = []
            for lines in repaired:
                for ln in lines:
                    for tok in ln.split():
                        cw = self._word_core(tok).lower()
                        if len(cw) >= self.word_min_len:
                            corpus_words.append(cw)
            if corpus_words:
                char_model = self._build_char_model(corpus_words)
                distinct = list(set(corpus_words))
                scores = np.array([self._char_score(w, char_model) for w in distinct])
                med = float(np.median(scores))
                mad = float(np.median(np.abs(scores - med))) or float(scores.std())
                char_thr = med - self.char_z * (mad if mad > 1e-9 else 1.0)

        # Index "substantial" lines per document.
        doc_lines: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        for ci, (chunk, lines) in enumerate(zip(chunks, repaired)):
            for li, ln in enumerate(lines):
                if len(ln.strip()) >= self.min_line_chars:
                    doc_lines[chunk.doc_id].append((ci, li, ln))

        cut: Dict[Tuple[int, int], bool] = {}
        line_override: Dict[Tuple[int, int], str] = {}

        for entries in doc_lines.values():
            survivors = []
            for ci, li, ln in entries:
                if self._is_garbled(ln):
                    cut[(ci, li)] = True
                else:
                    survivors.append((ci, li, ln))

            # LEVEL 2: keep the lines that mutually cohere.
            m = len(survivors)
            kept: List[Tuple[int, int, str, Optional[np.ndarray]]] = []
            core = None
            if m >= self.min_doc_lines and m >= 2:
                norm_texts = [self._norm(ln) for _, _, ln in survivors]
                emb = self._embed(norm_texts)
                sims = emb @ emb.T
                np.fill_diagonal(sims, -1.0)
                by_text: Dict[str, List[int]] = defaultdict(list)
                for idx, t in enumerate(norm_texts):
                    by_text[t].append(idx)
                for idxs in by_text.values():
                    if len(idxs) > 1:
                        sims[np.ix_(idxs, idxs)] = -1.0
                k = min(self.knn, m - 1)
                density = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
                keep = self._keep_mask(density)
                rows = []
                for (ci, li, ln), row, kp in zip(survivors, emb, keep):
                    if kp:
                        kept.append((ci, li, ln, row))
                        rows.append(row)
                    else:
                        cut[(ci, li)] = True
                if rows:
                    c = np.mean(rows, axis=0)
                    nrm = np.linalg.norm(c)
                    core = c / nrm if nrm > 0 else None
            else:
                kept = [(ci, li, ln, None) for ci, li, ln in survivors]

            # LEVEL 3: clean alien words inside the kept lines.
            if self.clean_words and char_model is not None and kept:
                vocab: Counter = Counter()
                for _, _, ln in survivors:
                    for tok in ln.split():
                        cw = self._word_core(tok).lower()
                        if len(cw) >= self.word_min_len:
                            vocab[cw] += 1
                for ci, li, ln, row in kept:
                    cleaned = self._clean_sentence(ln, vocab, char_model, char_thr, core, row)
                    if cleaned != ln:
                        line_override[(ci, li)] = cleaned

        # Reassemble each chunk.
        kept_chunks: List[Chunk] = []
        dropped_chunks = 0
        cut_lines = 0
        for ci, (chunk, lines) in enumerate(zip(chunks, repaired)):
            new_lines: List[str] = []
            for li, ln in enumerate(lines):
                if cut.get((ci, li)):
                    cut_lines += 1
                    continue
                new_lines.append(line_override.get((ci, li), ln))
            new_text = "\n".join(new_lines)
            if len(new_text.strip()) < self.min_keep_chars:
                dropped_chunks += 1
                continue
            kept_chunks.append(
                Chunk(
                    id=chunk.id,
                    doc_id=chunk.doc_id,
                    text=new_text,
                    tokens=chunk.tokens,
                    metadata=chunk.metadata,
                )
            )

        logger.info(
            "SelfConsistencyCleaner: kept %d / %d chunks (dropped %d empty; "
            "cut %d dirt lines; cleaned %d lines at word level; %d docs)",
            len(kept_chunks), n, dropped_chunks, cut_lines,
            len(line_override), len(doc_lines),
        )
        return kept_chunks
