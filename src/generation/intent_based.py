"""IntentBasedGenerator — prepend user intents to corpus documents.

Documentation says *what a function is*; users search for *what they are trying
to do*. Embedding a doc that never uses the user's vocabulary makes it hard to
retrieve. This generator closes that gap without rewriting the documentation:

  stage 1 (LLM)   — read a documentation fragment, ask for ``n_intents``
                    distinct PURPOSES the described function serves, phrased the
                    way a user would state the goal ("drop duplicate rows before
                    aggregating"), not the way the docs describe the API.
  stage 2 (local) — prepend those intents to the ORIGINAL document text. No
                    second LLM call: the body is the untouched corpus text, only
                    with query-shaped anchors in front of it.

So 1 source document -> 1 augmented document. Cost is one LLM call per document,
and the factual content is exactly the corpus's — nothing is invented, which is
the point compared with the rewrite-style generators.

Typically run with ``keep_source_docs: False`` so the index holds the augmented
versions instead of both (otherwise every document has a near-duplicate).

Reads the CORPUS only (never benchmark data), so it is a deployable-style
augmentation rather than an oracle.

Prompts are declared through ``PromptDiscoverable``: the policy prompts are
optimizable, the output-format contract is frozen because ``_parse_intents``
depends on it. See src/agent_constructor/prompts.py and §7-§8 of the overview.

NB: no ``from __future__ import annotations`` — keep annotations concrete for
the registry's dependency introspection.
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
from src.agent_constructor.prompts import PromptDiscoverable
from src.utils import DOCUMENT_SRC_DOCUMENTS
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)

_INTENT_SPLIT = re.compile(r"^\s*#{0,3}\s*INTENT\s*\d*\s*#*\s*$", re.M | re.I)


class IntentBasedGenerator(PromptDiscoverable, Generator):
    """Prepend LLM-generated user intents to each corpus document.

    Args:
        url: base URL of the OpenAI-compatible LLM endpoint.
        model_name: model id to call.
        prob: fraction of input documents to process (subsample).
        n_intents: purposes requested per document (default 5).
        num_workers: parallel LLM calls at build time.
        input_source: corpus source to read (default 'documents').
        output_source: ``source`` tag for produced docs (default = input_source,
            so the chunker treats them exactly like the originals).
        temperature: sampling temperature.
        intent_header: line introducing the prepended block.
        max_fragment_chars: how much of a document the LLM sees in stage 1.
        keep_unaugmented: emit the original text when stage 1 yields nothing,
            so a failed call never silently shrinks the corpus.
        seed: RNG seed for the subsample.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        prob: float = 0.05,
        n_intents: int = 5,
        num_workers: int = 4,
        input_source: str = DOCUMENT_SRC_DOCUMENTS,
        output_source: Optional[str] = None,
        temperature: float = 0.7,
        intent_header: str = "Common questions this answers:",
        max_fragment_chars: int = 1600,
        keep_unaugmented: bool = True,
        seed: int = 7,
        name: str = "intent_based_generator",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.prob = max(0.0, min(1.0, prob))
        self.n_intents = max(1, int(n_intents))
        self.num_workers = max(1, int(num_workers))
        self.input_source = input_source
        self.output_source = output_source or input_source
        self.temperature = float(temperature)
        self.intent_header = intent_header
        self.max_fragment_chars = int(max_fragment_chars)
        self.keep_unaugmented = bool(keep_unaugmented)
        self.seed = int(seed)

    # -------------------------------------------------------------------- LLM
    def _chat(self, system: Text, user: Text) -> Text:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------- stage 1: intents
    def _intents(self, doc: Document) -> List[Text]:
        lib = (doc.metadata or {}).get("library") or "the library"
        fragment = (doc.text or "")[: self.max_fragment_chars]

        system = self.prompt("intent_system", (
            "You are a senior Python engineer who knows why practitioners reach "
            "for a given API."
        ))
        main = self.render_prompt("intent_main", (
            "Below is a fragment of $lib documentation.\n"
            "-----\n$fragment\n-----\n"
            "List $n DIFFERENT practical PURPOSES this documentation serves — "
            "the real end goals a user has in mind when they need it, phrased as "
            "the question that user would type into a search box, in their own "
            "vocabulary rather than the documentation's. Each must be one line, "
            "concrete, and clearly different from the others. Do NOT write code "
            "and do NOT restate the signature.\n"
        ), lib=lib, fragment=fragment, n=self.n_intents)
        tail = self.prompt("intent_tail", (
            "Format STRICTLY as:\n### INTENT 1\n<one line>\n"
            "### INTENT 2\n<one line>\n..."
        ), optimizable=False)

        return self._parse_intents(self._chat(system, main + tail))

    def _parse_intents(self, text: Text) -> List[Text]:
        parts = _INTENT_SPLIT.split(text or "")
        candidates = parts[1:] if len(parts) > 1 else parts
        intents = []
        for part in candidates:
            line = part.strip().splitlines()[0].strip() if part.strip() else ""
            line = line.lstrip("-•* ").strip()
            if line and line not in intents:
                intents.append(line)
        return intents[: self.n_intents]

    # ---------------------------------------------------- stage 2: prepending
    def _augment(self, doc: Document, intents: List[Text]) -> Text:
        """Query-shaped anchors in front of the untouched corpus text."""
        if not intents:
            return doc.text or ""
        block = "\n".join(f"- {intent}" for intent in intents)
        return f"{self.intent_header}\n{block}\n\n{(doc.text or '').strip()}\n"

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        ids_offset = len(documents) + 1
        rng = random.Random(self.seed)

        pool = [doc for doc in documents if doc.source == self.input_source]
        if not pool:
            logger.warning("IntentBasedGenerator: no '%s' documents in the corpus",
                           self.input_source)
            return []
        pool = rng.sample(pool, max(1, round(len(pool) * self.prob)))

        tracker = get_active()
        harvested = {}
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as ex:
            futures = {ex.submit(self._intents, doc): doc for doc in pool}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{self.name}:intents"):
                doc = futures[fut]
                try:
                    harvested[doc.id] = fut.result()
                except Exception as exc:   # one bad call must not kill the run
                    logger.warning("intent generation failed for %s: %s",
                                   doc.id, exc)
                    harvested[doc.id] = []

        synth_docs: List[Document] = []
        n_augmented = 0
        n_intents = 0
        for doc in pool:
            intents = harvested.get(doc.id) or []
            if not intents and not self.keep_unaugmented:
                continue
            n_augmented += bool(intents)
            n_intents += len(intents)

            metadata = dict(doc.metadata or {})
            metadata.update({
                "section": "intent_based_gen",
                "generated_from": doc.id,
                "intents": intents,
            })
            synth_docs.append(
                Document(
                    id=str(len(synth_docs) + ids_offset),
                    text=self._augment(doc, intents),
                    source=self.output_source,
                    metadata=metadata,
                )
            )

        logger.info("IntentBasedGenerator: %d docs -> %d augmented (%d with "
                    "intents, %d intents total, %d workers)",
                    len(pool), len(synth_docs), n_augmented, n_intents,
                    self.num_workers)
        return synth_docs
