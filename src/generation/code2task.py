"""CodeToTaskGenerator — reformulate corpus code examples into task/solution pairs.

For each code example drawn from the corpus (``input_source``, default the
``examples`` table), ask the LLM for ``n_variants`` distinct *problem statements*
whose answer would be that code. Each variant becomes a new document whose body is
``task formulation + the original code`` — the same code reused across ``n_variants``
new examples with different phrasings. So 1 example -> ``n_variants`` new examples,
and the produced content is task+code (not just code).

Unlike the paraphraser/code2doc oracles, this reads the CORPUS (not the DS1000
reference answers), so it is a deployable-style augmentation, not an oracle.

Typically run with ``keep_source_docs: False`` at the pipeline level so the index
contains only these reworked examples (the original examples are used as input and
then dropped). Output ``source`` defaults to 'examples' so each task+code unit is
a single atomic chunk (RecursiveChunker does not split ``source=='examples'``).

NB: no ``from __future__ import annotations`` — keep annotations concrete for the
registry's dependency introspection.
"""

import logging
import random
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


class CodeToTaskGenerator(Generator):
    """Generate ``n_variants`` task formulations per corpus code example.

    Args:
        url: base URL of the OpenAI-compatible LLM endpoint.
        model_name: model id to call.
        prob: fraction of input examples to process (subsample, as in codeeval).
        n_variants: number of task formulations per example (default 3).
        num_workers: parallel LLM calls at build time.
        input_source: which corpus source to read (default 'examples').
        output_source: ``source`` tag for produced docs (default 'examples').
        temperature: sampling temperature for phrasing diversity.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        prob: float = 0.05,
        n_variants: int = 3,
        num_workers: int = 4,
        input_source: str = DOCUMENT_SRC_EXAMPLES,
        output_source: str = DOCUMENT_SRC_EXAMPLES,
        temperature: float = 0.7,
        name: str = "code2task_generator",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.prob = max(0.0, min(1.0, prob))
        self.n_variants = max(1, int(n_variants))
        self.num_workers = max(1, int(num_workers))
        self.input_source = input_source
        self.output_source = output_source
        self.temperature = float(temperature)

    # -------------------------------------------------------------------- LLM
    def _formulate(self, code: Text) -> List[Text]:
        n = self.n_variants
        user = (
            "Below is a self-contained Python code snippet that is the SOLUTION to "
            "some programming task.\n\n"
            "# Code\n" + (code or "") + "\n\n"
            f"Write {n} DIFFERENT problem statements (task formulations) for which "
            "this exact code would be a correct answer. Vary the wording and framing; "
            "each must be a clear, self-contained task description in English. Do NOT "
            "include any code or solution — only the task text.\n\n"
            f"Return exactly {n} variants, each preceded by its own marker line:\n"
            "### VARIANT 1\n<task text only>\n### VARIANT 2\n<task text only>\n..."
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert at writing programming exercises."},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        text = response.choices[0].message.content or ""
        return self._parse_variants(text, n)

    def _parse_variants(self, text: Text, n: int) -> List[Text]:
        parts = _VARIANT_SPLIT.split(text)
        candidates = parts[1:] if len(parts) > 1 else parts
        variants = [c.strip() for c in candidates if c.strip()]
        if not variants and text.strip():
            variants = [text.strip()]
        return variants[:n]

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        ids_offset = len(documents) + 1

        pool = [doc for doc in documents if doc.source == self.input_source]
        pool = random.sample(pool, round(len(pool) * self.prob))

        def work(doc: Document):
            return doc, self._formulate(doc.text)

        results = []
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as ex:
            futures = [ex.submit(work, d) for d in pool]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{self.name}:formulate"):
                try:
                    results.append(fut.result())
                except Exception as exc:  # one bad example must not kill the run
                    logger.warning("code2task example failed: %s", exc)

        synth_docs: List[Document] = []
        for doc, variants in results:
            for order, task_text in enumerate(variants):
                body = ("# Task\n" + task_text.strip()
                        + "\n\n# Solution\n" + (doc.text or "").strip()).strip()
                synth_docs.append(
                    Document(
                        id=str(len(synth_docs) + ids_offset),
                        text=body,
                        source=self.output_source,
                        metadata={
                            "generated_from": doc.id,
                            "order_id": order,
                        },
                    )
                )
        logger.info("CodeToTaskGenerator: %d examples -> %d task/solution docs "
                    "(%d workers)", len(pool), len(synth_docs), self.num_workers)
        return synth_docs
