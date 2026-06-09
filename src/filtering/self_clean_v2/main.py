"""SelfConsistencyCleanerV2 (F1_v2) — self_clean adapted to REALISTIC dirt.

Same "text is its own reference" cleaner as `self_clean`, with cheap text
NORMALISERS in front (HTML strip; mojibake fix), and — important now that code
examples are merged into chunks — a CODE/TEXT split: **code lines are kept
verbatim** and excluded from the text-cleaning levels (L1 symbols, L2 line
coherence, L3 alien words). The word/char/semantic cleaning runs on TEXT only.

This class is the BASE for:
  * F2.1 / F2.2 — override `_handle_code` to drop chunks with bad code (AST) or
    repair code via an LLM;
  * F3.1 / F3.2 — also override `_document_allowed` to drop off-topic documents.

NB: do NOT add `from __future__ import annotations` — the registry injects
`embedder` by matching the `Agent` type, which breaks under string annotations.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_LETTER_RUN = re.compile(r"[A-Za-z]+")
_JUNK = "#$%^&*~`"
_JC = re.escape(_JUNK)
_L1_RE = re.compile(rf"(?<=[A-Za-z])[{_JC}]|[{_JC}](?=[A-Za-z])")

_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_ENTITY = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
           "&quot;": '"', "&#39;": "'", "&apos;": "'"}
_ENT_ANY = re.compile(r"&[#\w]+;")
_MOJI_CHARS = "ÃÂâ€ÐÑðÒ"

# A line looks like Python code (REPL prompt, keyword, assignment, call, ...).
_CODE_RE = re.compile(
    r"""^\s*(
        >>>|\.\.\.|
        import\s|from\s.+\simport|
        def\s|class\s|return(\s|$)|raise\s|yield\s|assert\s|
        @\w+|
        for\s.+:|while\s.+:|if\s.+:|elif\s.+:|else\s*:|try\s*:|except|finally\s*:|with\s.+:|
        \w+\s*=\s*\S|
        \w+(\.\w+)*\s*\(|
        print\(|plt\.|np\.|pd\.|tf\.|torch\.
    )""",
    re.VERBOSE,
)


class SelfConsistencyCleanerV2(Filter):
    """F1_v2: realistic-dirt-aware self-consistency cleaner (text; code kept)."""

    def __init__(
        self,
        embedder: Agent,
        min_line_chars: int = 12,
        strip_html: bool = True,
        fix_mojibake: bool = True,
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
        name: str = "self_clean_v2_filter",
    ) -> None:
        self.embedder = embedder
        self.min_line_chars = int(min_line_chars)
        self.strip_html_flag = bool(strip_html)
        self.fix_mojibake_flag = bool(fix_mojibake)
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

    # ---- normalisers ----------------------------------------------------

    def _strip_html(self, line: str) -> str:
        line = _TAG.sub(" ", line)
        for k, v in _ENTITY.items():
            line = line.replace(k, v)
        return _ENT_ANY.sub(" ", line)

    @staticmethod
    def _weird(s: str) -> int:
        return sum(ch in _MOJI_CHARS for ch in s)

    def _fix_mojibake(self, line: str) -> str:
        if not any(ch in line for ch in ("Ã", "Â", "â€", "Ð", "Ñ", "ð")):
            return line
        try:
            fixed = line.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return line
        return fixed if self._weird(fixed) < self._weird(line) else line

    def _normalise(self, line: str) -> str:
        if self.fix_mojibake_flag:
            line = self._fix_mojibake(line)
        if self.strip_html_flag:
            line = self._strip_html(line)
        return self._l1_repair(line)

    # ---- shared helpers -------------------------------------------------

    @staticmethod
    def _norm(line: str) -> str:
        return _WS.sub(" ", line.strip().lower())

    def _l1_repair(self, line: str) -> str:
        return _L1_RE.sub("", line)

    def _is_garbled(self, line: str) -> bool:
        s = line.strip()
        compact = s.replace(" ", "")
        if not compact:
            return False
        alpha_ratio = sum(c.isalpha() for c in compact) / len(compact)
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
        low_idx, high_idx = order[:best_t], order[best_t:]
        mu_high = float(density[high_idx].mean())
        std_high = float(density[high_idx].std())
        mu_low = float(density[low_idx].mean())
        thr = mu_high - self.sep_z * std_high
        keep = np.ones(m, dtype=bool)
        if mu_low < thr:
            keep = density >= thr
        return keep

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
        tg, total, vocab = model
        s = "^" + w + "$"
        if len(s) < 3:
            return 0.0
        lps = []
        denom = total + vocab
        for i in range(len(s) - 2):
            lps.append(math.log((tg.get(s[i:i + 3], 0) + 1) / denom))
        lps.sort()
        k = min(2, len(lps))
        return sum(lps[:k]) / k

    def _build_char_model_from(self, text_lines: List[List[str]]):
        words: List[str] = []
        for lines in text_lines:
            for ln in lines:
                for tok in ln.split():
                    cw = self._word_core(tok).lower()
                    if len(cw) >= self.word_min_len:
                        words.append(cw)
        if not words:
            return None, None
        model = self._build_char_model(words)
        distinct = list(set(words))
        scores = np.array([self._char_score(w, model) for w in distinct])
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med))) or float(scores.std())
        thr = med - self.char_z * (mad if mad > 1e-9 else 1.0)
        return model, thr

    def _clean_sentence(self, line, vocab, model, char_thr, core, line_emb) -> str:
        toks = line.split()
        content = []
        for j, tok in enumerate(toks):
            cw = self._word_core(tok)
            if len(cw) >= self.word_min_len:
                content.append((j, cw.lower(), any(ord(c) > 127 for c in cw)))
        if len(content) < self.min_sentence_words:
            return line
        cand = [c for c in content if c[2] or vocab[c[1]] <= 1]
        if not cand:
            return line
        remove, semantic_cand = set(), []
        for j, low, foreign in cand:
            if self._char_score(low, model) < char_thr:
                remove.add(j)
            else:
                semantic_cand.append((j, low))
        if (self.semantic_word_check and core is not None
                and line_emb is not None and semantic_cand):
            base = float(line_emb @ core)
            variants = [" ".join(t for k, t in enumerate(toks) if k != j)
                        for j, _ in semantic_cand]
            ve = self._embed(variants)
            for (j, _), v in zip(semantic_cand, ve):
                if float(v @ core) - base > self.word_improve:
                    remove.add(j)
        if not remove:
            return line
        max_cut = max(1, int(self.max_word_cut_frac * len(content)))
        if len(remove) > max_cut:
            remove = set(sorted(remove)[:max_cut])
        return " ".join(t for k, t in enumerate(toks) if k not in remove)

    # ---- code/text split + extension hooks -----------------------------

    def _is_code_line(self, line: str) -> bool:
        return bool(_CODE_RE.search(line)) if line.strip() else False

    def _document_allowed(self, doc_id, lines) -> bool:
        """F3.x overrides: drop whole off-topic documents. Base keeps all."""
        return True

    def _handle_code(self, chunks, repaired, code_idx, cut, drop_chunk, override) -> None:
        """F2.x overrides: act on code lines (drop chunk / LLM-fix). Base: keep."""
        return

    # ---- main -----------------------------------------------------------

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        n = len(chunks)
        if n == 0:
            return []

        repaired: List[List[str]] = [
            [self._normalise(ln) for ln in (c.text or "").split("\n")]
            for c in chunks
        ]

        # code/text classification — code lines are protected from text-cleaning
        code_idx: Dict[int, Set[int]] = {
            ci: {li for li, ln in enumerate(lines) if self._is_code_line(ln)}
            for ci, lines in enumerate(repaired)
        }

        char_model = char_thr = None
        if self.clean_words:
            text_only = [
                [ln for li, ln in enumerate(lines) if li not in code_idx[ci]]
                for ci, lines in enumerate(repaired)
            ]
            char_model, char_thr = self._build_char_model_from(text_only)

        doc_lines: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        doc_all: Dict[str, List[str]] = defaultdict(list)
        for ci, (chunk, lines) in enumerate(zip(chunks, repaired)):
            doc_all[chunk.doc_id].extend(lines)
            for li, ln in enumerate(lines):
                if li in code_idx[ci]:           # protect code
                    continue
                if len(ln.strip()) >= self.min_line_chars:
                    doc_lines[chunk.doc_id].append((ci, li, ln))

        cut: Dict[Tuple[int, int], bool] = {}
        override: Dict[Tuple[int, int], str] = {}
        drop_chunk: Set[int] = set()
        dropped_docs = set()

        for doc_id, entries in doc_lines.items():
            if not self._document_allowed(doc_id, doc_all[doc_id]):   # STEP 0 (F3.x)
                dropped_docs.add(doc_id)
                for ci, li, _ in entries:
                    cut[(ci, li)] = True
                continue

            survivors = []
            for ci, li, ln in entries:
                if self._is_garbled(ln):
                    cut[(ci, li)] = True
                else:
                    survivors.append((ci, li, ln))

            m = len(survivors)
            kept = survivors
            core = None
            kept_rows = {}
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
                kept, rows = [], []
                for (ci, li, ln), row, kp in zip(survivors, emb, keep):
                    if kp:
                        kept.append((ci, li, ln))
                        kept_rows[(ci, li)] = row
                        rows.append(row)
                    else:
                        cut[(ci, li)] = True
                if rows:
                    c = np.mean(rows, axis=0)
                    nrm = np.linalg.norm(c)
                    core = c / nrm if nrm > 0 else None

            if self.clean_words and char_model is not None and kept:
                vocab: Counter = Counter()
                for _, _, ln in survivors:
                    for tok in ln.split():
                        cw = self._word_core(tok).lower()
                        if len(cw) >= self.word_min_len:
                            vocab[cw] += 1
                for ci, li, ln in kept:
                    cleaned = self._clean_sentence(
                        ln, vocab, char_model, char_thr, core, kept_rows.get((ci, li)))
                    if cleaned != ln:
                        override[(ci, li)] = cleaned

        # CODE handling hook (F2.x): may fill drop_chunk / override / cut.
        self._handle_code(chunks, repaired, code_idx, cut, drop_chunk, override)

        kept_chunks: List[Chunk] = []
        dropped_chunks = 0
        for ci, (chunk, lines) in enumerate(zip(chunks, repaired)):
            if ci in drop_chunk:
                dropped_chunks += 1
                continue
            new_lines = []
            for li, ln in enumerate(lines):
                if cut.get((ci, li)):
                    continue
                new_lines.append(override.get((ci, li), ln))
            new_text = "\n".join(new_lines)
            if len(new_text.strip()) < self.min_keep_chars:
                dropped_chunks += 1
                continue
            kept_chunks.append(Chunk(id=chunk.id, doc_id=chunk.doc_id, text=new_text,
                                     tokens=chunk.tokens, metadata=chunk.metadata))

        logger.info(
            "SelfConsistencyCleanerV2: kept %d / %d chunks (dropped %d; "
            "off-topic docs %d; %d docs)",
            len(kept_chunks), n, dropped_chunks, len(dropped_docs), len(doc_lines),
        )
        return kept_chunks
