"""Code-aware cleaners (F2.1, F2.2) — extend F1_v2 with a CODE path.

Both subclass SelfConsistencyCleanerV2: text lines are cleaned by the inherited
"text is its own reference" levels; CODE lines (kept verbatim by the base) are
handled here.

  * F2.1 `CodeAwareSelectCleaner` — drop a whole CHUNK if its code is bad
    (won't parse). No LLM. No-op on prose-only corpora -> universal.
  * F2.2 `CodeAwareLLMCleaner` — REPAIR bad code with an LLM, giving it the WHOLE
    document as context. LLM only on chunks whose code is bad (cheap gate); if
    the fix fails, drop the chunk (never keep broken code).

NB: no `from __future__ import annotations`; `embedder: Agent` must be annotated
on each concrete __init__ (registry injects by type). No **kwargs in __init__
(registry treats it as a required parameter).
"""

import ast
import logging
import re
import textwrap
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from tqdm import tqdm

from src.agent_constructor.agent import Agent
from src.filtering.self_clean_v2.main import SelfConsistencyCleanerV2
from src.utils.token_tracker import get_active, set_active


logger = logging.getLogger(__name__)
_FENCE = re.compile(r"^\s*```[\w+-]*\s*$", re.MULTILINE)


def _strip_repl(line: str) -> str:
    s = line.lstrip()
    if s[:4] in (">>> ", "... "):
        return s[4:]
    if s in (">>>", "..."):
        return ""
    return line


class _CodeMixin:
    """Shared bad-code detection."""

    bad_code_ratio: float = 0.5

    def _block_ok(self, code_lines: List[str]) -> bool:
        with warnings.catch_warnings():          # parsing dirty code spams SyntaxWarning
            warnings.simplefilter("ignore")
            body = textwrap.dedent("\n".join(_strip_repl(l) for l in code_lines))
            try:
                ast.parse(body)
                return True
            except (SyntaxError, ValueError):
                pass
            lines = [_strip_repl(l).strip() for l in code_lines if l.strip()]
            if not lines:
                return True
            fails = 0
            for l in lines:
                try:
                    ast.parse(l)
                except (SyntaxError, ValueError):
                    fails += 1
            return fails / len(lines) <= self.bad_code_ratio


class CodeAwareSelectCleaner(_CodeMixin, SelfConsistencyCleanerV2):
    """F2.1 — F1_v2 + drop chunks whose code does not parse."""

    def __init__(self, embedder: Agent, bad_code_ratio: float = 0.5,
                 name: str = "code_select_filter") -> None:
        super().__init__(embedder=embedder, name=name)
        self.bad_code_ratio = float(bad_code_ratio)

    def _handle_code(self, chunks, repaired, code_idx, cut, drop_chunk,
                     override, dropped_docs) -> None:
        dropped = 0
        for ci, lines in enumerate(repaired):
            if chunks[ci].doc_id in dropped_docs:      # off-topic doc -> already gone
                continue
            cidx = code_idx.get(ci)
            if not cidx:
                continue
            code_lines = [lines[li] for li in sorted(cidx)]
            if not self._block_ok(code_lines):
                drop_chunk.add(ci)
                dropped += 1
        logger.info("CodeAwareSelectCleaner: dropped %d chunks with bad code", dropped)


class CodeAwareLLMCleaner(_CodeMixin, SelfConsistencyCleanerV2):
    """F2.2 — F1_v2 + LLM repairs bad code (whole document as context)."""

    def __init__(self, embedder: Agent, llm_url: str = None, llm_model: str = None,
                 bad_code_ratio: float = 0.5, max_context_chars: int = 4000,
                 num_workers: int = 4, name: str = "code_llm_filter") -> None:
        super().__init__(embedder=embedder, name=name)
        self.bad_code_ratio = float(bad_code_ratio)
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.max_context_chars = int(max_context_chars)
        self.num_workers = int(num_workers)
        self._client = None

    def _llm(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.llm_url, api_key="vllm")
        return self._client

    def _fix_code(self, doc_text: str, code: str) -> str:
        try:
            resp = self._llm().chat.completions.create(
                model=self.llm_model,
                temperature=0.1,
                max_tokens=700,
                messages=[
                    {"role": "system", "content":
                     "You repair corrupted Python code from documentation "
                     "(broken encoding, leaked HTML, typo'd characters). Return "
                     "ONLY the corrected code — no prose, no markdown fences."},
                    {"role": "user", "content":
                     f"Document (context):\n{doc_text[:self.max_context_chars]}\n\n"
                     f"Corrupted code block from this document — return it fixed:\n{code}"},
                ],
            )
            out = resp.choices[0].message.content or ""
            return _FENCE.sub("", out).strip()
        except Exception as exc:                       # noqa: BLE001
            logger.warning("CodeAwareLLMCleaner: LLM fix failed (%s)", exc)
            return ""

    def _handle_code(self, chunks, repaired, code_idx, cut, drop_chunk,
                     override, dropped_docs) -> None:
        doc_text = defaultdict(list)
        for ci, ch in enumerate(chunks):
            doc_text[ch.doc_id].append(ch.text or "")
        doc_text = {d: "\n".join(v) for d, v in doc_text.items()}

        # 1) SERIAL cheap AST gate -> worklist of broken-code chunks.
        #    `_block_ok` uses warnings.catch_warnings() (NOT thread-safe), so the
        #    gate stays serial; only the network LLM calls below are parallel.
        work = []   # (ci, order, code_str)
        for ci, lines in enumerate(repaired):
            if chunks[ci].doc_id in dropped_docs:      # off-topic doc -> already gone
                continue
            cidx = code_idx.get(ci)
            if not cidx:
                continue
            order = sorted(cidx)
            code_lines = [lines[li] for li in order]
            if self._block_ok(code_lines):
                continue
            work.append((ci, order, "\n".join(code_lines)))

        # 2) PARALLEL LLM repair (network-bound). Workers re-arm the thread-local
        #    token tracker so the filter's token usage is still counted.
        fixes: List[str] = [""] * len(work)
        if work:
            self._llm()                                # pre-warm client (avoid init race)
            tracker = get_active()
            with ThreadPoolExecutor(max_workers=self.num_workers,
                                    initializer=set_active,
                                    initargs=(tracker,)) as ex:
                fut_to_i = {
                    ex.submit(self._fix_code,
                              doc_text.get(chunks[ci].doc_id, ""), code): i
                    for i, (ci, _order, code) in enumerate(work)
                }
                for fut in tqdm(as_completed(fut_to_i), total=len(fut_to_i),
                                desc=f"{self.name}:code-fix"):
                    fixes[fut_to_i[fut]] = fut.result()

        # 3) SERIAL validate + apply (AST gate again -> serial).
        fixed_n = dropped_n = 0
        for (ci, order, _code), fixed in zip(work, fixes):
            if fixed and self._block_ok(fixed.split("\n")):
                override[(ci, order[0])] = fixed
                for li in order[1:]:
                    cut[(ci, li)] = True
                fixed_n += 1
            else:
                drop_chunk.add(ci)
                dropped_n += 1
        logger.info("CodeAwareLLMCleaner: fixed %d, dropped %d code chunks "
                    "(%d candidates, %d workers)",
                    fixed_n, dropped_n, len(work), self.num_workers)
