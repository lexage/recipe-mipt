"""Corpus-grounded how-to recipe generator (exp17, methods 1 and 2).

Rationale, from the exp16 ablation:

  * ``howto_*`` recipes carried +9.10 of the +10.15 total. ``migr_*`` carried
    -0.20 and ``guide_*`` +2.00 through retrieval — so this generator writes
    recipes only. Format rules belong in the SOLVER's system prompt, where the
    same text was worth +5.90 instead of +2.00.
  * The manual augmentation drew its topics from observed DS-1000 failures,
    which is the leak this generator is built to avoid. Topics come from an API
    census over the corpus instead: whatever the documentation actually talks
    about, weighted by how often it appears.
  * Retrieval coverage decided everything in exp16 (46.8% of tasks for recipes
    vs 7.2% for migration notes). Recipes are therefore phrased as questions,
    matching the shape of an incoming query rather than the shape of reference
    documentation.

Two methods share this class:

  ``context_docs = 0``   METHOD 1. The model sees an API name and writes from
                         parametric memory. Cheap, and the baseline to beat.
  ``context_docs > 0``   METHOD 2. The prompt additionally carries real corpus
                         snippets mentioning that API — documentation prose and
                         example code. The model paraphrases what the corpus
                         actually says instead of recalling it.

Neither method ever reads DS-1000. T1-style leakage (task literals copied into
documents) is impossible here by construction, not by instruction.
"""

import json
import logging
import os
import re
import sqlite3
import textwrap

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from src.agent_constructor.core import Document
from src.agent_constructor.generator import Generator
from src.agent_constructor.prompts import PromptDiscoverable
from src.utils import DOCUMENT_SRC_DOCUMENTS
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)

_TITLE_RE = re.compile(r"^TITLE:\s*(.+)$", re.M)
_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)
_QUAL_RE = re.compile(r"\b((?:np|pd|plt|sns|torch|tf|scipy|sklearn|sp)"
                      r"(?:\.[A-Za-z_][A-Za-z0-9_]*){1,3})\s*\(")
_METH_RE = re.compile(r"\.([a-z_][a-z0-9_]{2,})\s*\(")
_METH_STOP = {"append", "format", "join", "split", "strip", "print", "range",
              "len", "get", "keys", "items", "values", "copy", "read", "write"}
_PREFIX_LIB = {"np": "numpy", "pd": "pandas", "plt": "matplotlib",
               "sns": "matplotlib", "torch": "pytorch", "tf": "tensorflow",
               "scipy": "scipy", "sp": "scipy", "sklearn": "sklearn"}


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def _strip_thinking(text: str) -> str:
    """Remove a reasoning preamble if the model emitted one.

    Qwen3 turns thinking on by default in its chat template, so responses arrive
    wrapped in <think>...</think>. ``enable_thinking`` switches that off at the
    request level; this is the second line of defence, for servers that ignore
    the flag or models that leak the tag anyway. Without it the parser either
    rejects nearly everything or files the reasoning as document text, and a
    model comparison turns into a comparison of parser luck.
    """
    text = _THINK_RE.sub("", text or "")
    # An unclosed tag means the reasoning ran into the token limit; nothing
    # after it is usable.
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def _slug(title: str, prefix: str) -> str:
    words = re.findall(r"[a-z0-9]+", title.lower())[:7]
    return prefix + "_" + "_".join(words) if words else prefix + "_doc"


def _code_ok(code: str) -> bool:
    """Accept a block only if it parses — as a script or as a function body."""
    for candidate in (code, "def _wrap_():\n" + textwrap.indent(code, "    ")):
        try:
            compile(candidate, "<gen>", "exec")
            return True
        except SyntaxError:
            continue
    return False


class HowtoRecipeGenerator(PromptDiscoverable, Generator):
    """Write short how-to recipes for the APIs the corpus actually uses.

    Args:
        url: base URL of the OpenAI-compatible endpoint for the GENERATOR model.
        model_name: generator model id (independent of the solver's model).
        path_to_db: corpus sqlite path; the ``examples`` table holds most of the
            code, so the census and the grounding snippets both need it.
        n_docs: target number of recipes. Keep this equal across methods —
            otherwise a comparison measures corpus size, not method quality.
        min_count: ignore APIs seen fewer than this many times in the census.
        context_docs: 0 selects METHOD 1; a positive value selects METHOD 2 and
            sets how many corpus snippets are pasted into each prompt.
        context_chars: per-snippet truncation, to keep prompts affordable.
        num_workers: parallel generation calls.
        temperature: sampling temperature for the generator.
        limit: cap the number of calls (smoke runs). 0 = no cap.
        max_doc_chars: drop recipes longer than this (chunker budget).
        dump_path: append surviving recipes to this jsonl for audit.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        path_to_db: str = "",
        n_docs: int = 200,
        min_count: int = 5,
        context_docs: int = 0,
        context_chars: int = 700,
        num_workers: int = 8,
        temperature: float = 0.7,
        limit: int = 0,
        max_doc_chars: int = 1100,
        dump_path: str = "",
        enable_thinking: Optional[bool] = None,
        use_system_role: bool = True,
        max_tokens: int = 700,
        name: str = "howto_recipe_generator",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.path_to_db = path_to_db
        self.n_docs = int(n_docs)
        self.min_count = int(min_count)
        self.context_docs = int(context_docs)
        self.context_chars = int(context_chars)
        self.num_workers = max(1, int(num_workers))
        self.temperature = float(temperature)
        self.limit = int(limit)
        self.max_doc_chars = int(max_doc_chars)
        self.dump_path = dump_path
        # None -> send nothing and let the server decide; False is what the
        # Qwen3 configs set, so their output is comparable with Qwen2.5's.
        self.enable_thinking = enable_thinking
        # Explicit ceiling: without it a reasoning model that ignores
        # enable_thinking can run to the context limit on every call.
        self.max_tokens = int(max_tokens)
        # Gemma's chat template has no system role and rejects the
        # message outright; folding the text into the user turn keeps
        # the instruction identical across model families.
        self.use_system_role = bool(use_system_role)
        self.stats = Counter()
        self._snippets = {}          # api token -> [corpus snippet, ...]

    @property
    def method(self) -> str:
        return "m2_grounded" if self.context_docs > 0 else "m1_parametric"

    # ------------------------------------------------------------------- LLM
    def _chat(self, user: str) -> str:
        extra = {}
        if self.enable_thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        system = self.prompt("chat_system", (
                    "You are a senior Python engineer writing short, precise "
                    "knowledge-base documents for a retrieval system."
                ))
        if self.use_system_role:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
        else:
            messages = [{"role": "user", "content": system + "\n\n" + user}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **({"extra_body": extra} if extra else {}),
        )
        return _strip_thinking(response.choices[0].message.content or "")

    # ---------------------------------------------------------------- census
    def _corpus_text(self, documents: List[Document]) -> List[Tuple[str, str]]:
        """(text, library) over documents plus the examples table."""
        rows = [(d.text, (d.metadata or {}).get("library")) for d in documents]
        if not self.path_to_db:
            return rows
        try:
            conn = sqlite3.connect(f"file:{self.path_to_db}?mode=ro", uri=True)
            rows += conn.execute(
                "SELECT e.content, l.name FROM examples e "
                "JOIN documents d ON e.doc_id = d.id "
                "JOIN sections s ON d.section_id = s.id "
                "JOIN libraries l ON s.library_id = l.id").fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("examples census failed (%s): %s",
                           self.path_to_db, exc)
        return rows

    def _census(self, corpus: List[Tuple[str, str]]) -> Counter:
        counts: Counter = Counter()
        for text, lib in corpus:
            text = text or ""
            for match in _QUAL_RE.finditer(text):
                token = match.group(1)
                root = token.split(".")[0]
                counts[(_PREFIX_LIB.get(root, root), token)] += 1
            if lib:
                for match in _METH_RE.finditer(text):
                    method = match.group(1)
                    if method not in _METH_STOP:
                        counts[(lib, "." + method)] += 1
        return counts

    def _select(self, counts: Counter) -> List[Tuple[str, str, int]]:
        """Top APIs, with per-library quotas proportional to census share."""
        counts = Counter({k: v for k, v in counts.items() if v >= self.min_count})
        if not counts:
            return []
        share = Counter()
        for (lib, _), n in counts.items():
            share[lib] += n
        total = sum(share.values())

        selected = []
        for lib, lib_total in share.items():
            quota = max(1, round(self.n_docs * lib_total / total))
            ranked = sorted(((tok, n) for (l, tok), n in counts.items() if l == lib),
                            key=lambda x: -x[1])
            selected += [(lib, tok, n) for tok, n in ranked[:quota]]
        selected.sort(key=lambda x: -x[2])
        return selected[:self.n_docs]

    # ------------------------------------------------------------- grounding
    def _collect_snippets(self, corpus: List[Tuple[str, str]],
                          apis: List[Tuple[str, str, int]]) -> None:
        """METHOD 2 only: gather real corpus passages mentioning each API."""
        wanted = {tok for _, tok, _ in apis}
        found = {tok: [] for tok in wanted}
        for text, _ in corpus:
            text = text or ""
            for token in wanted:
                if len(found[token]) >= self.context_docs:
                    continue
                position = text.find(token)
                if position == -1:
                    continue
                # A window around the mention, snapped to line boundaries so the
                # model never sees a fragment cut mid-statement.
                start = max(0, position - self.context_chars // 3)
                end = min(len(text), position + self.context_chars)
                window = text[start:end]
                window = window[window.find("\n") + 1:] if start else window
                found[token].append(window.strip())
        self._snippets = found
        self.stats["grounded_apis"] = sum(1 for v in found.values() if v)
        self.stats["grounding_snippets"] = sum(len(v) for v in found.values())

    # ----------------------------------------------------------------- jobs
    def _jobs(self, apis: List[Tuple[str, str, int]]) -> List[Tuple[str, str, str]]:
        jobs = []
        for lib, token, _ in apis:
            common = (
                f"Write ONE short how-to document about `{token}` ({lib}) for a "
                f"retrieval index.\n"
                "Structure:\n"
                "  TITLE: a question a developer would type, in their own words "
                f"('How do I ...?'), naming `{token}` or what it does.\n"
                "  Then 2-4 sentences: the modern recommended approach and the "
                "mistake people actually make.\n"
                "  Then a minimal runnable example, 2-8 lines, using your own "
                "tiny invented data.\n"
                "Under 120 words plus the code. Invent variable names yourself; "
                "use neutral ones (a, df, x). Format STRICTLY as:\n"
                "TITLE: <the question>\n<text, code in ``` fences>"
            )
            snippets = self._snippets.get(token) or []
            if snippets:
                # METHOD 2: ground the recipe in what the corpus says. The model
                # is told to paraphrase, not to lift — copied prose would just
                # duplicate documents the index already contains.
                reference = "\n\n".join(
                    f"--- corpus passage {i + 1} ---\n{s[:self.context_chars]}"
                    for i, s in enumerate(snippets))
                prompt = (
                    f"Below are real passages from a {lib} documentation corpus "
                    f"that mention `{token}`.\n\n{reference}\n\n"
                    "Use them for factual accuracy: correct argument names, "
                    "correct return types, current API spelling. Do NOT copy "
                    "their wording or reuse their variable and column names — "
                    "write the document in your own words with your own data.\n\n"
                    + common
                )
                self.stats["jobs_grounded"] += 1
            else:
                prompt = common
                self.stats["jobs_plain"] += 1
            jobs.append((lib, "howto", prompt))
        return jobs

    # ------------------------------------------------------------- parsing
    def _parse(self, text: str, lib: str, prefix: str):
        text = (text or "").strip()
        if not text:
            self.stats["drop_empty"] += 1
            return None
        match = _TITLE_RE.search(text)
        title = match.group(1).strip() if match else text.splitlines()[0][:70]
        body = _TITLE_RE.sub("", text, count=1).strip()
        content = title + "\n\n" + body

        blocks = _FENCE_RE.findall(content)
        if any(not _code_ok(b) for b in blocks):
            self.stats["drop_bad_code"] += 1
            return None
        content = _FENCE_RE.sub(
            lambda m: textwrap.indent(m.group(1).rstrip(), "    ") + "\n",
            content).strip() + "\n"

        if len(content) < 120:
            self.stats["drop_too_short"] += 1
            return None
        if len(content) > self.max_doc_chars:
            self.stats["drop_too_long"] += 1
            return None
        self.stats["parsed_ok"] += 1
        return (lib, _slug(title, prefix), content)

    # ----------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        self.stats = Counter()
        corpus = self._corpus_text(documents)
        counts = self._census(corpus)
        self.stats["census_apis"] = len(counts)
        apis = self._select(counts)
        if not apis:
            logger.warning("HowtoRecipeGenerator: empty API census")
            return []
        if self.limit:
            apis = apis[:self.limit]
        self.stats["selected_apis"] = len(apis)

        if self.context_docs > 0:
            logger.info("%s: collecting grounding snippets for %d APIs",
                        self.name, len(apis))
            self._collect_snippets(corpus, apis)

        logger.info("%s (%s): %d recipe jobs over %d distinct APIs",
                    self.name, self.method, len(apis), len(counts))
        parsed = self._run_jobs(self._jobs(apis))
        return self._package(parsed, documents)

    def _run_jobs(self, jobs):
        def work(job):
            lib, prefix, prompt = job
            return self._parse(self._chat(prompt), lib, prefix)

        parsed = []
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as pool:
            futures = [pool.submit(work, j) for j in jobs]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"{self.name}:{self.method}"):
                try:
                    doc = future.result()
                    if doc is not None:
                        parsed.append(doc)
                except Exception as exc:  # one bad call must not kill the run
                    self.stats["drop_call_error"] += 1
                    logger.warning("recipe generation call failed: %s", exc)
        return parsed

    def _package(self, parsed, documents: List[Document]) -> List[Document]:
        seen = set()
        offset = len(documents) + 1
        synth = []
        dump = None
        if self.dump_path:
            os.makedirs(os.path.dirname(self.dump_path) or ".", exist_ok=True)
            dump = open(self.dump_path, "w", encoding="utf-8")
        try:
            for lib, doc_name, content in parsed:
                if doc_name in seen:
                    self.stats["drop_dup_name"] += 1
                    continue
                seen.add(doc_name)
                synth.append(Document(
                    id=str(len(synth) + offset),
                    text=content,
                    source=DOCUMENT_SRC_DOCUMENTS,
                    metadata={"library": lib, "section": "howto_gen",
                              "doc_name": doc_name, "method": self.method},
                ))
                if dump:
                    dump.write(json.dumps(
                        {"library": lib, "name": doc_name, "content": content,
                         "method": self.method, "model": self.model_name},
                        ensure_ascii=False) + "\n")
        finally:
            if dump:
                dump.close()
        logger.info("%s: %d documents kept | %s",
                    self.name, len(synth), dict(self.stats))
        return synth
