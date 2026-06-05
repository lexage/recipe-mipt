"""SelfConsistencyCleaner — clean text using the text ITSELF as the reference.

Our NEW mechanism (научная новизна): every known filter compares text to
something EXTERNAL — corpus frequencies (dedup / IDF), a trained classifier,
a target distribution (DSIR), big-model perplexity, or an LLM judge (see the
2025 report, §3.1). We instead ask, at every granularity: *does this piece fit
the piece that contains it?* Injected dirt is a foreign body — it falls out of
its own host (its word, its sentence, its document). We keep the self-consistent
core and CUT what does not belong; a chunk is dropped only if nothing survives.

This is a CLEANER, not a selector, and it works at THREE granularities, each
with its OWN reference (not the whole corpus, not even the whole document):

  * LEVEL 1 — a word vs itself (cheap, NON-semantic). Real tokens are letter
    runs, digit runs or identifiers (letters/digits/`_`/`-`). Junk symbols
    inserted INTO a word (e.g. "ar#r@ay") are stripped when glued to letters,
    so "ar#r@ay" -> "array" while code (`np.array(`), numbers and domain terms
    (`int64`, `float32`, `read_csv`) are left intact. Letter/digit corruption
    without symbols ("arr4ay", "arxray") is indistinguishable from real terms
    here, so it is left for level 2 to judge by meaning.

  * LEVEL 2 — a word vs its SENTENCE (semantic). The sentence is its own
    reference: we measure how close each word is to the rest of its sentence
    and drop the words that sit NOTICEABLY farther from their sentence than the
    other words do (a relative, per-sentence cut). Foreign-language and random
    swaps are handled here purely by meaning — nothing is removed just for being
    non-English. We do NOT use the document core: with up to ~3/4 dirt the
    document is mostly noise, so the sentence is the only trustworthy anchor.

  * LEVEL 3 — a line vs the LINES THAT BELONG TOGETHER (semantic). We do NOT
    take a document mean (with most lines dirty the mean is dirt). Instead we
    ask which lines mutually cohere: each line's neighbour-density (agreement
    with its closest DISTINCT lines) tells whether it is part of a meaningful
    group or an outlier. The cut is relative — derived from the document's own
    density distribution — and VERBATIM-duplicate neighbours are discounted (by
    exact text, not by embedding similarity) so repeated junk can't vouch for
    its own copies, while genuinely similar good lines still vouch normally.

NB: do NOT add `from __future__ import annotations` here — the registry injects
`embedder` by matching the `Agent` type, which breaks under string annotations.
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_LETTER_RUN = re.compile(r"[A-Za-z]+")
# Junk symbols stripped at level 1 ONLY when glued to a letter (so space-
# separated code operators like `a % b` and dots in `np.array` survive).
_JUNK = "#$%^&*~`!@"
_JC = re.escape(_JUNK)
_L1_RE = re.compile(rf"(?<=[A-Za-z])[{_JC}]|[{_JC}](?=[A-Za-z])")


class SelfConsistencyCleaner(Filter):
    """Clean dirt that does not fit its own word / sentence / document.

    Args:
        embedder: text->vector model, injected by the pipeline builder (the
            `embedder` component). Used for levels 2 and 3.
        min_line_chars: lines shorter than this are kept as-is (too short to
            judge — e.g. headings, single tokens).
        garbled_alpha_ratio, garbled_max_run: a line that is mostly non-letters
            with no real word-run is symbol soup -> dropped (cheap pre-clean).
        knn: neighbours used for the level-3 density estimate.
        sep_z: level-3 relative margin — a line is cut only if its density sits
            more than `sep_z` core-std-devs below the coherent group.
        min_doc_lines: documents with fewer substantial lines skip level 3.
        clean_words: enable level 2 (per-word, semantic, sentence-anchored).
        word_min_len: minimum letter-length of a "content word".
        min_sentence_words: a line needs at least this many content words to run
            level 2 (too few -> can't judge a word against its sentence).
        sep_z_word: level-2 relative margin — a candidate word is cut only if its
            closeness to the rest of its sentence sits more than `sep_z_word`
            below the sentence's typical word closeness.
        max_word_cut_frac: never remove more than this fraction of a sentence's
            words (safety against gutting a real sentence).
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
        sep_z_word: float = 1.5,
        max_word_cut_frac: float = 0.34,
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
        self.sep_z_word = float(sep_z_word)
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
        """Level-3 relative cut from the document's own density distribution."""
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

    def _clean_sentence(self, line: str, vocab: Counter) -> str:
        """Level 2: drop words that don't fit their own sentence (semantic).

        A word is a removal CANDIDATE only if it is non-recurring in the
        document or written in a foreign script (cheap trigger — narrows the
        work, does NOT decide). The DECISION is semantic: keep the word only if
        it is not a clear outlier vs the rest of its sentence.
        """
        toks = line.split()
        # content words with their token index
        content = []
        for j, tok in enumerate(toks):
            core = self._word_core(tok)
            if len(core) >= self.word_min_len:
                foreign = any(ord(c) > 127 for c in core)
                content.append((j, tok, core.lower(), foreign))
        if len(content) < self.min_sentence_words:
            return line

        candidates = [
            c for c in content if c[3] or vocab[c[2]] <= 1   # foreign or non-recurring
        ]
        if not candidates:
            return line

        # closeness of each content word to its WHOLE sentence (sentence anchor)
        sent_norm = self._norm(line)
        word_texts = [c[1] for c in content]
        emb = self._embed([sent_norm] + word_texts)
        e_sent = emb[0]
        coh = emb[1:] @ e_sent                 # cosine of each word to the sentence

        med = float(np.median(coh))
        mad = float(np.median(np.abs(coh - med)))
        spread = mad if mad > 1e-6 else float(coh.std())
        if spread <= 1e-6:
            return line
        thr = med - self.sep_z_word * spread

        cand_idx = {c[0] for c in candidates}
        order = np.argsort(coh)                # lowest closeness first
        max_cut = int(self.max_word_cut_frac * len(content))
        remove_tok_idx = set()
        for pos in order:
            if len(remove_tok_idx) >= max_cut:
                break
            if coh[pos] >= thr:
                break                          # rest are fine
            j = content[pos][0]
            if j in cand_idx:                  # only candidates may be removed
                remove_tok_idx.add(j)

        if not remove_tok_idx:
            return line
        return " ".join(t for j, t in enumerate(toks) if j not in remove_tok_idx)

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

        # Index "substantial" lines per document (a doc spans several chunks).
        doc_lines: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        for ci, (chunk, lines) in enumerate(zip(chunks, repaired)):
            for li, ln in enumerate(lines):
                if len(ln.strip()) >= self.min_line_chars:
                    doc_lines[chunk.doc_id].append((ci, li, ln))

        cut: Dict[Tuple[int, int], bool] = {}
        line_override: Dict[Tuple[int, int], str] = {}   # level-2 cleaned lines

        for entries in doc_lines.values():
            # cheap pre-clean: drop symbol-soup lines.
            survivors = []
            for ci, li, ln in entries:
                if self._is_garbled(ln):
                    cut[(ci, li)] = True
                else:
                    survivors.append((ci, li, ln))

            # LEVEL 3: keep the lines that mutually cohere.
            m = len(survivors)
            kept = survivors
            if m >= self.min_doc_lines and m >= 2:
                norm_texts = [self._norm(ln) for _, _, ln in survivors]
                emb = self._embed(norm_texts)
                sims = emb @ emb.T
                np.fill_diagonal(sims, -1.0)
                # discount VERBATIM duplicates (boilerplate/filler copies) so
                # they can't vouch for one another; good lines have distinct
                # text and keep all their real neighbours.
                by_text: Dict[str, List[int]] = defaultdict(list)
                for idx, t in enumerate(norm_texts):
                    by_text[t].append(idx)
                for idxs in by_text.values():
                    if len(idxs) > 1:
                        sims[np.ix_(idxs, idxs)] = -1.0
                k = min(self.knn, m - 1)
                density = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
                keep = self._keep_mask(density)
                kept = []
                for (ci, li, ln), keep_it in zip(survivors, keep):
                    if keep_it:
                        kept.append((ci, li, ln))
                    else:
                        cut[(ci, li)] = True

            # LEVEL 2: clean alien words inside the kept lines.
            if self.clean_words and kept:
                vocab: Counter = Counter()
                for _, _, ln in survivors:
                    for tok in ln.split():
                        core = self._word_core(tok)
                        if len(core) >= self.word_min_len:
                            vocab[core.lower()] += 1
                for ci, li, ln in kept:
                    cleaned = self._clean_sentence(ln, vocab)
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
