"""OracleIntentGenerator — ORACLE generation: mine a table of user-intent phrasings
from DS1000 reference answers, then bake the k nearest intents into EVERY corpus
document/example with a cheap embedding KNN classifier (no LLM at corpus scale).

Oracle counterpart to :class:`src.generation.intent_based.IntentBasedGenerator`
(deployable): that generator invents intents per corpus DOCUMENT via one LLM call
each, reading the corpus only, and can afford just a ~5% subsample because every
document costs an LLM call. This generator instead:

  stage 1 (LLM, ~len(dataset) calls)  — mine ``n_intents`` short purpose-phrasings
      per DS1000 task from its (prompt, reference_code) — the test answers, which
      is what makes this an oracle. Flattened one-row-per-intent and cached to
      ``intents_cache_path`` (a jsonl "table"), so the LLM stage runs once no
      matter how many pipeline configs / k values reuse it.
  stage 2 (embeddings only, no LLM) — embed every input Document (regardless of
      its 'documents' / 'examples' source tag) and every mined intent with the
      pipeline's own embedder, then for each document pick its ``k`` nearest
      intents by cosine similarity (a KNN classifier) and prepend them to its
      body, exactly like IntentBasedGenerator's prepend format.

Because stage 2 has no LLM cost, this can run over the FULL corpus instead of a
subsample — but note it only sees whatever the DB component already loaded, so
to bake into both tables the DB component needs ``index_sources: ["documents",
"examples"]`` (default is documents-only).

Oracle because stage 1 reads the DS1000 reference answers (test set) — an upper
bound / diagnostic on how much perfectly-matched intent vocabulary could help,
not a deployable method. See src/generation/paraphrase.py for the sibling
oracle experiment and src/generation/intent_based.py for the deployable pair.

NB: no ``from __future__ import annotations`` — keep annotations concrete for
the registry's dependency introspection.
"""

import gzip
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Document, Text
from src.agent_constructor.generator import Generator
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)

_INTENT_SPLIT = re.compile(r"^\s*#{0,3}\s*INTENT\s*\d*\s*#*\s*$", re.M | re.I)


class OracleIntentGenerator(Generator):
    """Mine intents from DS1000 answers, KNN-bake the top-k into every document.

    Args:
        embedder: text->vector model, injected by the pipeline builder.
        url: base URL of the OpenAI-compatible LLM endpoint (stage 1 only).
        model_name: model id to call for stage 1.
        dataset_path: path to the DS1000 ``.jsonl.gz`` (source of the intents).
        intents_cache_path: where the mined intents table is cached (jsonl, one
            row per intent). Reused across runs unless ``rebuild_cache``.
        rebuild_cache: force re-mining stage 1 even if the cache file exists.
        n_intents: purposes mined per DS1000 task (default 5).
        k: intents baked into each corpus document (default 3).
        min_similarity: if set, an intent below this cosine similarity is not
            baked in even if it's in the top-k (can leave a document with < k
            intents, or none). None = always attach exactly k.
        intent_header: line introducing the prepended block.
        max_fragment_chars: chars of prompt/reference_code shown to the LLM
            in stage 1.
        max_doc_chars: chars of each document embedded in stage 2.
        embed_batch_size: documents/intents per embedder call.
        compare_batch_size: rows per matmul block when finding neighbours.
        num_workers: parallel LLM calls in stage 1.
        temperature: sampling temperature for stage 1.
        limit: if > 0, only mine intents for the first ``limit`` DS1000 tasks
            (smoke runs). 0 = all.
    """

    def __init__(
        self,
        embedder: Agent,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset_path: str = "data/ds1000/ds1000.jsonl.gz",
        intents_cache_path: str = "data/oracle_intents.jsonl",
        rebuild_cache: bool = False,
        n_intents: int = 5,
        k: int = 3,
        min_similarity: Optional[float] = None,
        intent_header: str = "Common questions this answers:",
        max_fragment_chars: int = 800,
        max_doc_chars: int = 2000,
        embed_batch_size: int = 32,
        compare_batch_size: int = 512,
        num_workers: int = 4,
        temperature: float = 0.7,
        limit: int = 0,
        name: str = "oracle_intent_generator",
    ):
        super().__init__(name)
        self.embedder = embedder
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.intents_cache_path = intents_cache_path
        self.rebuild_cache = bool(rebuild_cache)
        self.n_intents = max(1, int(n_intents))
        self.k = max(1, int(k))
        self.min_similarity = min_similarity
        self.intent_header = intent_header
        self.max_fragment_chars = int(max_fragment_chars)
        self.max_doc_chars = int(max_doc_chars)
        self.embed_batch_size = int(embed_batch_size)
        self.compare_batch_size = int(compare_batch_size)
        self.num_workers = max(1, int(num_workers))
        self.temperature = float(temperature)
        self.limit = int(limit)

    # ------------------------------------------------------ stage 1: mining
    def _load_tasks(self) -> List[dict]:
        tasks = []
        with gzip.open(self.dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                tasks.append(json.loads(line))
        if self.limit and self.limit < len(tasks):
            tasks = tasks[: self.limit]
        return tasks

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

    def _intents_for_task(self, task: dict) -> List[Text]:
        meta = task.get("metadata", {}) or {}
        lib = meta.get("library") or "the library"
        prompt = (task.get("prompt") or "")[: self.max_fragment_chars]
        code = (task.get("reference_code") or "")[: self.max_fragment_chars]

        system = (
            "You are a senior Python engineer who knows why practitioners reach "
            "for a given API."
        )
        user = (
            f"Below is a {lib} programming task and its reference solution.\n"
            f"-----\n# Task\n{prompt}\n\n# Reference solution\n{code}\n-----\n"
            f"List {self.n_intents} DIFFERENT practical PURPOSES this solution "
            "serves — the real end goals a user has in mind when they need it, "
            "phrased as the question that user would type into a search box, in "
            "their own vocabulary rather than the task's. Each must be one line, "
            "concrete, and clearly different from the others. Do NOT write code "
            "and do NOT restate the task verbatim.\n"
            "Format STRICTLY as:\n### INTENT 1\n<one line>\n"
            "### INTENT 2\n<one line>\n..."
        )
        return self._parse_intents(self._chat(system, user))

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

    def _mine_intents_table(self) -> List[dict]:
        tasks = self._load_tasks()

        def work(task: dict):
            return task, self._intents_for_task(task)

        results = []
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as ex:
            futures = [ex.submit(work, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{self.name}:mine-intents"):
                try:
                    results.append(fut.result())
                except Exception as exc:   # one bad task must not kill the run
                    logger.warning("oracle intent mining failed for a task: %s", exc)

        rows = []
        for task, intents in results:
            meta = task.get("metadata", {}) or {}
            for order, intent in enumerate(intents):
                rows.append({
                    "problem_id": meta.get("problem_id"),
                    "library": meta.get("library"),
                    "perturbation_type": meta.get("perturbation_type"),
                    "order_id": order,
                    "intent": intent,
                    "reference_code": task.get("reference_code", ""),
                })

        os.makedirs(os.path.dirname(self.intents_cache_path) or ".", exist_ok=True)
        with open(self.intents_cache_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("OracleIntentGenerator: mined %d intents from %d tasks -> %s",
                    len(rows), len(tasks), self.intents_cache_path)
        return rows

    def _load_or_build_intents_table(self) -> List[dict]:
        if not self.rebuild_cache and os.path.isfile(self.intents_cache_path):
            rows = []
            with open(self.intents_cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            logger.info("OracleIntentGenerator: loaded %d cached intents from %s",
                        len(rows), self.intents_cache_path)
            return rows
        return self._mine_intents_table()

    # ---------------------------------------------------- stage 2: KNN baking
    def _embed_all(self, texts: List[Text]) -> np.ndarray:
        out: List[np.ndarray] = []
        bs = self.embed_batch_size
        pbar = tqdm(range(0, len(texts), bs), desc=f"{self.name}:embed")
        for start in pbar:
            batch = texts[start:start + bs]
            vectors = self.embedder.run(batch)
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            out.append(arr)
        return np.concatenate(out, axis=0)

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def _augment(self, doc_text: Text, intents: List[Text]) -> Text:
        if not intents:
            return doc_text or ""
        block = "\n".join(f"- {intent}" for intent in intents)
        return f"{self.intent_header}\n{block}\n\n{(doc_text or '').strip()}\n"

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        rows = self._load_or_build_intents_table()
        if not rows or not documents:
            logger.warning("OracleIntentGenerator: nothing to do "
                           "(%d intents, %d documents)", len(rows), len(documents))
            return []

        intent_texts = [r["intent"] for r in rows]
        intent_vecs = self._normalize(self._embed_all(intent_texts))

        doc_texts = [(d.text or "")[: self.max_doc_chars] for d in documents]
        doc_vecs = self._normalize(self._embed_all(doc_texts))

        ids_offset = len(documents) + 1
        synth_docs: List[Document] = []
        n = len(documents)
        bs = self.compare_batch_size
        k = min(self.k, len(rows))
        n_baked = 0
        n_intents_used = 0

        for start in tqdm(range(0, n, bs), desc=f"{self.name}:knn-bake"):
            end = min(start + bs, n)
            block = doc_vecs[start:end]                      # (b, d)
            sims = block @ intent_vecs.T                      # (b, n_intents)
            top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]

            for local_i in range(end - start):
                doc = documents[start + local_i]
                row_sims = sims[local_i]
                idxs = top_idx[local_i]
                idxs = idxs[np.argsort(-row_sims[idxs])]       # best first
                if self.min_similarity is not None:
                    idxs = [j for j in idxs if row_sims[j] >= self.min_similarity]
                chosen = [rows[j]["intent"] for j in idxs]

                metadata = dict(doc.metadata or {})
                metadata.update({
                    "section": "oracle_intent_gen",
                    "generated_from": doc.id,
                    "oracle_intents": chosen,
                })
                synth_docs.append(
                    Document(
                        id=str(len(synth_docs) + ids_offset),
                        text=self._augment(doc.text, chosen),
                        source=doc.source,
                        metadata=metadata,
                    )
                )
                n_baked += bool(chosen)
                n_intents_used += len(chosen)

        logger.info("OracleIntentGenerator: %d documents -> %d baked (%d with "
                    "intents, %d intents attached total, table size %d, k=%d)",
                    n, len(synth_docs), n_baked, n_intents_used, len(rows), k)
        return synth_docs
