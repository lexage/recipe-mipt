"""ParaphraseGenerator — ORACLE generation: paraphrase DS1000 reference answers.

Oracle experiment (exp6): for each DS1000 task, paraphrase its ``reference_code``
into ``n_paraphrases`` functionally-equivalent-but-not-identical code variants
and inject them into the corpus as ``examples`` documents. They get chunked (one
chunk each, via RecursiveChunker's ``source == 'examples'`` branch), embedded
into the vector index by ``add_chunks``, and are then found at retrieval purely
by vector similarity to the task — no parent-document attachment needed.

This is an ORACLE / ceiling: it peeks at the test answers, so it is NOT a
deployable method. Its purpose is twofold:
  1. upper bound — how much can ideal corpus content move PASS@1;
  2. diagnostic — does the RAG pipeline actually exploit corpus content? If even
     paraphrases of the answers in the corpus do not lift PASS@1, retrieval /
     context assembly is broken (the exp4 failure mode).

Pipeline component: plugs into the ``generator`` slot of SIMPLE / SIMPLE_WITH_
DOC_FILTER. Build-time cost is one LLM call per task; calls run in parallel
(``num_workers``) and re-arm the thread-local token tracker so usage is counted.

NB: no ``from __future__ import annotations`` — keep annotations concrete for the
registry's dependency introspection.
"""

import gzip
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from openai import OpenAI
from tqdm import tqdm

from src.agent_constructor.core import Document, Text
from src.agent_constructor.generator import Generator
from src.utils import DOCUMENT_SRC_EXAMPLES
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)

_VARIANT_SPLIT = re.compile(r"^\s*#{0,3}\s*VARIANT\s*\d+\s*#*\s*$", re.M | re.I)
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*\n|\n\s*```\s*$")


def _strip_fences(code: str) -> str:
    code = code.strip()
    # remove a leading ```python / trailing ``` fence if present
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else ""
    if code.rstrip().endswith("```"):
        code = code.rstrip()[: code.rstrip().rfind("```")]
    return code.strip()


class ParaphraseGenerator(Generator):
    """Generate ``n_paraphrases`` paraphrases of each DS1000 reference answer.

    Args:
        url: base URL of the OpenAI-compatible LLM endpoint.
        model_name: model id to call.
        dataset_path: path to the DS1000 ``.jsonl.gz`` (source of tasks/answers).
        n_paraphrases: number of variants per task (default 3).
        num_workers: parallel LLM calls at build time (default 4).
        limit: if > 0, only the first ``limit`` tasks (smoke). 0 = all.
        temperature: sampling temperature for variant diversity.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset_path: str = "data/ds1000/ds1000.jsonl.gz",
        n_paraphrases: int = 3,
        num_workers: int = 4,
        limit: int = 0,
        temperature: float = 0.6,
        name: str = "paraphrase_generator",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.n_paraphrases = max(1, int(n_paraphrases))
        self.num_workers = max(1, int(num_workers))
        self.limit = int(limit)
        self.temperature = float(temperature)

    # ------------------------------------------------------------------ tasks
    def _load_tasks(self) -> List[dict]:
        tasks = []
        with gzip.open(self.dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                tasks.append(json.loads(line))
        if self.limit and self.limit < len(tasks):
            tasks = tasks[: self.limit]
        return tasks

    # -------------------------------------------------------------------- LLM
    def _paraphrase(self, prompt: Text, reference_code: Text) -> List[Text]:
        n = self.n_paraphrases
        user = (
            "Below is a Python programming problem and a reference solution.\n\n"
            "# Problem\n" + (prompt or "") + "\n\n"
            "# Reference solution\n" + (reference_code or "") + "\n\n"
            f"Write {n} alternative solutions that are FUNCTIONALLY EQUIVALENT to "
            "the reference solution but written differently — vary names, style, "
            "or approach. Each must be valid, self-contained Python. Do NOT copy "
            "the reference verbatim and do NOT add explanations or prose.\n\n"
            f"Return exactly {n} variants, each preceded by its own marker line:\n"
            "### VARIANT 1\n<code only>\n### VARIANT 2\n<code only>\n..."
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional Python developer."},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        text = response.choices[0].message.content or ""
        return self._parse_variants(text, n)

    def _parse_variants(self, text: Text, n: int) -> List[Text]:
        parts = _VARIANT_SPLIT.split(text)
        # When markers are present, parts[0] is the preamble before the first
        # "### VARIANT" marker — drop it; parts[1:] are the variant bodies.
        candidates = parts[1:] if len(parts) > 1 else parts
        variants = [c for c in (_strip_fences(p) for p in candidates) if c]
        # If the model ignored the markers, fall back to the whole response.
        if not variants:
            stripped = _strip_fences(text)
            variants = [stripped] if stripped else []
        return variants[:n]

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        ids_offset = len(documents) + 1
        tasks = self._load_tasks()

        def work(task: dict):
            meta = task.get("metadata", {}) or {}
            variants = self._paraphrase(task.get("prompt", ""),
                                        task.get("reference_code", ""))
            return meta, variants

        results = []
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as ex:
            futures = [ex.submit(work, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{self.name}:paraphrase"):
                try:
                    results.append(fut.result())
                except Exception as exc:  # one bad task must not kill the run
                    logger.warning("paraphrase task failed: %s", exc)

        synth_docs: List[Document] = []
        for meta, variants in results:
            for order, code in enumerate(variants):
                synth_docs.append(
                    Document(
                        id=str(len(synth_docs) + ids_offset),
                        text=code,
                        source=DOCUMENT_SRC_EXAMPLES,
                        metadata={
                            "generated_from": meta.get("problem_id"),
                            "library": meta.get("library"),
                            "order_id": order,
                        },
                    )
                )
        logger.info("ParaphraseGenerator: %d tasks -> %d paraphrase examples "
                    "(%d workers)", len(tasks), len(synth_docs), self.num_workers)
        return synth_docs
