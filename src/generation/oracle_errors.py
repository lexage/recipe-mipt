"""Oracle generator (exp17, method 4): write recipes from observed failures.

This reproduces, inside the pipeline, what the external agent did by hand: read
the solver's answers next to the reference solutions and write documents that
would have prevented the failures. The point is a CAPABILITY CHECK — can the
local model extract anything generalizable from a correct answer at all, or was
that step relying on a much stronger model?

It is deliberately leaky. Anything it produces is scored on tasks the generator
was shown, so its number is an upper bound and a diagnostic, never a result to
publish. Two dossiers make the distinction explicit:

  all-1000   directly comparable to the manual +10.15, and equally tuned-on-test
  train-300  evaluated with ``--exclude-split``, so the number means something

Failures are batched by library rather than sent one at a time. Exp16 showed the
manual pass produced roughly one recipe per observed failure — 140 recipes for
140 distinct errors — which is exactly the shape that does not generalise.
Batching forces the model to look across several failures and name the pattern
they share, and it is also what keeps the document count near the 200 budget
instead of exploding to one per error.
"""

import json
import logging
import os
import re
import textwrap

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

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
_DOC_SPLIT_RE = re.compile(r"^---+$", re.M)


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
    for candidate in (code, "def _wrap_():\n" + textwrap.indent(code, "    ")):
        try:
            compile(candidate, "<gen>", "exec")
            return True
        except SyntaxError:
            continue
    return False


class OracleErrorGenerator(PromptDiscoverable, Generator):
    """Turn a failure dossier into retrieval documents.

    Args:
        url: base URL of the OpenAI-compatible endpoint for the GENERATOR model.
        model_name: generator model id.
        dossier_path: jsonl with one record per task — ``prompt``,
            ``reference_code``, ``model_code``, ``passed``, ``library``.
            Build it with ``db_scripts/make_e4_agent_dossier.py``.
        n_docs: target number of documents (keep equal to the other methods).
        batch_size: failures shown per call. Larger batches push the model
            toward the shared pattern instead of a per-task fix.
        failures_only: ignore tasks the solver already got right.
        num_workers, temperature, limit, max_doc_chars, dump_path: as usual.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        dossier_path: str = "",
        n_docs: int = 200,
        batch_size: int = 6,
        docs_per_batch: int = 3,
        failures_only: bool = True,
        num_workers: int = 8,
        temperature: float = 0.7,
        limit: int = 0,
        max_doc_chars: int = 1100,
        max_task_chars: int = 1200,
        dump_path: str = "",
        enable_thinking: Optional[bool] = None,
        use_system_role: bool = True,
        max_tokens: int = 1800,
        name: str = "oracle_error_generator",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.dossier_path = dossier_path
        self.n_docs = int(n_docs)
        self.batch_size = max(1, int(batch_size))
        self.docs_per_batch = max(1, int(docs_per_batch))
        self.failures_only = bool(failures_only)
        self.num_workers = max(1, int(num_workers))
        self.temperature = float(temperature)
        self.limit = int(limit)
        self.max_doc_chars = int(max_doc_chars)
        self.max_task_chars = int(max_task_chars)
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

    # ------------------------------------------------------------------- LLM
    def _chat(self, user: str) -> str:
        extra = {}
        if self.enable_thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        system = self.prompt("chat_system", (
                    "You are a senior Python engineer doing error analysis. You "
                    "write short, generalizable knowledge-base documents."
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

    # ---------------------------------------------------------------- input
    def _load(self):
        if not self.dossier_path or not os.path.isfile(self.dossier_path):
            raise FileNotFoundError(
                f"dossier not found: {self.dossier_path!r} — build it with "
                f"db_scripts/make_e4_agent_dossier.py")
        by_lib = defaultdict(list)
        with open(self.dossier_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.stats["dossier_records"] += 1
                if self.failures_only and record.get("passed"):
                    continue
                by_lib[record.get("library") or "unknown"].append(record)
        self.stats["failures"] = sum(len(v) for v in by_lib.values())
        return by_lib

    def _jobs(self, by_lib):
        """Batch failures per library; quota proportional to failure share."""
        total = sum(len(v) for v in by_lib.values()) or 1
        jobs = []
        for lib, records in sorted(by_lib.items()):
            quota = max(1, round(self.n_docs * len(records) / total))
            n_batches = max(1, round(quota / self.docs_per_batch))
            stride = max(1, len(records) // n_batches)
            for start in range(0, len(records), stride):
                batch = records[start:start + self.batch_size]
                if not batch:
                    continue
                cases = []
                for i, record in enumerate(batch, 1):
                    cases.append(
                        f"### case {i}\n"
                        f"TASK:\n{(record.get('prompt') or '')[:self.max_task_chars]}\n"
                        f"WHAT THE MODEL WROTE:\n{(record.get('model_code') or '').strip()}\n"
                        f"WHAT WAS CORRECT:\n{(record.get('reference_code') or '').strip()}\n")
                prompt = (
                    f"Below are {len(batch)} {lib} tasks the model got wrong, "
                    "each with its answer and the correct solution.\n\n"
                    + "\n".join(cases) +
                    f"\nFind what these failures have IN COMMON and write "
                    f"{self.docs_per_batch} short how-to documents that would "
                    "prevent this class of mistake on any similar task.\n\n"
                    "HARD RULES:\n"
                    "- Write about the general technique, never about these "
                    "specific tasks.\n"
                    "- Do NOT reuse any column name, variable name, string "
                    "literal or number from the tasks above. Invent neutral "
                    "data of your own (a, df, x).\n"
                    "- Do not mention grading, checkers, expected output or "
                    "hidden tests.\n"
                    "- A document that only helps on one of these tasks is "
                    "worthless; if you cannot find a shared pattern, write "
                    "fewer documents.\n\n"
                    "Format each document STRICTLY as:\n"
                    "TITLE: <a question a developer would type, 'How do I ...?'>\n"
                    "<2-4 sentences, then a 2-8 line example in ``` fences>\n"
                    "Separate documents with a line containing only ---"
                )
                jobs.append((lib, prompt))
                if len(jobs) * self.docs_per_batch >= self.n_docs:
                    break
        return jobs

    # -------------------------------------------------------------- parsing
    def _parse_many(self, text: str, lib: str):
        out = []
        for chunk in _DOC_SPLIT_RE.split(text or ""):
            chunk = chunk.strip()
            if not chunk or "TITLE:" not in chunk:
                continue
            match = _TITLE_RE.search(chunk)
            title = match.group(1).strip() if match else chunk.splitlines()[0][:70]
            body = _TITLE_RE.sub("", chunk, count=1).strip()
            content = title + "\n\n" + body

            blocks = _FENCE_RE.findall(content)
            if any(not _code_ok(b) for b in blocks):
                self.stats["drop_bad_code"] += 1
                continue
            content = _FENCE_RE.sub(
                lambda m: textwrap.indent(m.group(1).rstrip(), "    ") + "\n",
                content).strip() + "\n"
            if len(content) < 120:
                self.stats["drop_too_short"] += 1
                continue
            if len(content) > self.max_doc_chars:
                self.stats["drop_too_long"] += 1
                continue
            self.stats["parsed_ok"] += 1
            out.append((lib, _slug(title, "oracle"), content))
        return out

    # ------------------------------------------------------------ interface
    def generate(self, documents: List[Document]) -> List[Document]:
        self.stats = Counter()
        by_lib = self._load()
        if not by_lib:
            logger.warning("OracleErrorGenerator: no failures in dossier")
            return []
        jobs = self._jobs(by_lib)
        if self.limit:
            jobs = jobs[:self.limit]
        logger.info("%s: %d batches over %d failures (%d libraries)",
                    self.name, len(jobs), self.stats["failures"], len(by_lib))

        parsed = []
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as pool:
            futures = [pool.submit(lambda j: self._parse_many(self._chat(j[1]), j[0]), j)
                       for j in jobs]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=self.name):
                try:
                    parsed.extend(future.result())
                except Exception as exc:
                    self.stats["drop_call_error"] += 1
                    logger.warning("oracle generation call failed: %s", exc)

        return self._package(parsed, documents)

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
                if len(synth) >= self.n_docs:
                    self.stats["drop_over_budget"] += 1
                    continue
                seen.add(doc_name)
                synth.append(Document(
                    id=str(len(synth) + offset),
                    text=content,
                    source=DOCUMENT_SRC_DOCUMENTS,
                    metadata={"library": lib, "section": "oracle_gen",
                              "doc_name": doc_name, "method": "m4_oracle"},
                ))
                if dump:
                    dump.write(json.dumps(
                        {"library": lib, "name": doc_name, "content": content,
                         "method": "m4_oracle", "model": self.model_name},
                        ensure_ascii=False) + "\n")
        finally:
            if dump:
                dump.close()
        logger.info("%s: %d documents kept | %s",
                    self.name, len(synth), dict(self.stats))
        return synth
