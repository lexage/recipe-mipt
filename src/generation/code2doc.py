"""CodeToDocGenerator — ORACLE generation: paraphrase DS1000 reference answers
AND attach a natural-language description to each paraphrase.

This is a thin extension of :class:`ParaphraseGenerator` (exp6, oracle). For each
DS1000 task it produces ``n_paraphrases`` functionally-equivalent code variants
(exactly like the paraphraser), then for EACH variant it generates a short textual
description and bundles ``description + code`` into a single document body. So one
paraphrase -> one document, and there are N documents per task.

Bundling text+code in one body is a deliberate replacement for the ``merge``
mechanism: instead of relying on ``<example_N>`` placeholders being stitched at
query time, we pre-bake the description together with the code so the whole unit
is retrieved as one coherent chunk.

Like the paraphraser this is an ORACLE (it reads the test reference answers) —
an upper bound / diagnostic, not a deployable method.

Output ``source`` is configurable (``output_source``, default 'examples') so the
bundled unit stays a single atomic chunk (the RecursiveChunker does not split
``source=='examples'`` documents). NB: no ``from __future__ import annotations``
— keep annotations concrete for the registry's dependency introspection.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from tqdm import tqdm

from src.agent_constructor.core import Document, Text
from src.generation.paraphrase import ParaphraseGenerator, _strip_fences
from src.utils import DOCUMENT_SRC_EXAMPLES
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)


class CodeToDocGenerator(ParaphraseGenerator):
    """Paraphrase DS1000 reference answers, then describe each paraphrase.

    Args mirror :class:`ParaphraseGenerator`, plus:
        output_source: ``source`` tag for the produced documents (default
            'examples' so the bundled description+code stays a single chunk).
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
        output_source: str = DOCUMENT_SRC_EXAMPLES,
        name: str = "code2doc_generator",
    ):
        super().__init__(
            url=url,
            model_name=model_name,
            dataset_path=dataset_path,
            n_paraphrases=n_paraphrases,
            num_workers=num_workers,
            limit=limit,
            temperature=temperature,
            name=name,
        )
        self.output_source = output_source

    # -------------------------------------------------------------------- LLM
    def _describe(self, code: Text) -> Text:
        """Return a short natural-language description of a code snippet."""
        user = (
            "Below is a self-contained Python code snippet.\n\n"
            "# Code\n" + (code or "") + "\n\n"
            "Write a concise natural-language description (2-4 sentences) of what "
            "this code does: the task it solves, the key library operations used, "
            "and the expected result. Do NOT repeat the code and do NOT add a "
            "preamble like 'This code' more than once. Description only."
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a precise technical writer."},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        return _strip_fences(response.choices[0].message.content or "")

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        ids_offset = len(documents) + 1
        tasks = self._load_tasks()

        def work(task: dict):
            meta = task.get("metadata", {}) or {}
            variants = self._paraphrase(task.get("prompt", ""),
                                        task.get("reference_code", ""))
            described = [(code, self._describe(code)) for code in variants]
            return meta, described

        results = []
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as ex:
            futures = [ex.submit(work, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{self.name}:describe"):
                try:
                    results.append(fut.result())
                except Exception as exc:  # one bad task must not kill the run
                    logger.warning("code2doc task failed: %s", exc)

        synth_docs: List[Document] = []
        for meta, described in results:
            for order, (code, desc) in enumerate(described):
                body = (desc.strip() + "\n\n" + code.strip()).strip()
                synth_docs.append(
                    Document(
                        id=str(len(synth_docs) + ids_offset),
                        text=body,
                        source=self.output_source,
                        metadata={
                            "generated_from": meta.get("problem_id"),
                            "library": meta.get("library"),
                            "order_id": order,
                        },
                    )
                )
        logger.info("CodeToDocGenerator: %d tasks -> %d described docs (%d workers)",
                    len(tasks), len(synth_docs), self.num_workers)
        return synth_docs
