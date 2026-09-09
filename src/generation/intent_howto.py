"""IntentHowtoGenerator — SEMI-ORACLE generation: DS1000 conditions -> intents
-> howto documents.

Two-stage pipeline, deliberately split so each stage can be cached and rerun
independently:

  stage 1 (LLM, ~len(dataset) calls) — for every DS1000 task, read only its
      *condition* (the natural-language problem statement plus whatever setup
      code precedes the harness scaffold) and mine ``n_intents_per_task``
      general, reusable "how do I ...?" purposes a real developer would have.
      The reference solution is never read — that is what makes this a
      SEMI-oracle rather than the full oracle in
      :class:`src.generation.oracle_intent.OracleIntentGenerator` (which mines
      from ``reference_code``) or :class:`src.generation.paraphrase.
      ParaphraseGenerator` (which paraphrases it). It is still an oracle
      relative to :class:`src.generation.intent_based.IntentBasedGenerator`
      (deployable, corpus-only) because it looks at benchmark task text at
      all. Mined intents are cached to ``intents_cache_path`` (a jsonl table)
      so stage 2/3 can be rerun with different ``clusterize``/``n_clusters``
      without re-mining.

      DS1000 ships each task as one of two harness shapes — "Problem: ...\\nA:
      \\n<code>...BEGIN SOLUTION" (845 tasks) or "<setup code>\\n# SOLUTION
      START" (155, matplotlib) — the condition is everything before that
      marker; the reference answer lives in a separate field this stage never
      touches.

      By default only one task per ``perturbation_origin_id`` is mined
      (``dedup_by_origin``): DS1000's Surface/Semantic/Difficult-Rewrite
      variants restate the same underlying problem, so mining all 1000 would
      spend the LLM budget on near-duplicate conditions before stage 2 ever
      sees them.

  stage 2 (embeddings only, no LLM, optional) — when ``clusterize`` is set,
      embed every mined intent and group them so ONE howto is written per
      cluster instead of per intent. High-dimensional text embeddings have
      pairwise distances that concentrate (the curse of dimensionality: in
      high ambient dimension almost all pairwise distances become similar,
      so raw k-means/nearest-neighbour on the raw vectors clusters on noise).
      RAPTOR mitigates this with UMAP (src/rag/raptor/core/cluster_utils.py);
      this generator uses PCA for the same purpose — project onto the
      low-dimensional subspace the embeddings actually vary along before
      clustering — because PCA is pure numpy/scipy with no extra native
      dependency (umap-learn pulls in numba, which requires numpy<=2.4 and
      broke on this project's numpy 2.5). The representative of each cluster
      is picked back in the ORIGINAL normalized embedding space (the medoid
      closest to the cluster centroid) — the reduced coordinates are a
      clustering aid, not a similarity measure to trust for picking the one
      intent that speaks for the group. With ``clusterize: False`` every
      intent gets its own document.

  stage 3 (LLM, one call per intent/representative) — write a howto document
      in the Question -> Answer -> Caveat shape the manual ``howto_*`` corpus
      uses (see the "Anatomy of the How-Tos" analysis: 96% "How do I...?"
      titles, one code block, 99% end on prose). Reuses the shared editorial
      contract in ``src.generation.doc_contract`` (``DOC_CLASS_RECIPE``) so a
      generated document is held to exactly the same form checks as
      PlanGuideGenerator's / ApiGuideGenerator's recipes: one correct code
      block on pre-existing variables, no invented literal data, no harness
      markers, a question title. ``api_token`` is left empty (an intent is
      not anchored to one specific API the way a corpus-evidence recipe is),
      so the ``api_token_missing`` check never fires here.

NB: no ``from __future__ import annotations`` — keep annotations concrete for
the registry's dependency introspection.
"""

import gzip
import json
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Document, Text
from src.agent_constructor.generator import Generator
from src.agent_constructor.prompts import PromptDiscoverable
from src.generation.doc_contract import (DOC_CLASS_RECIPE, SELF_CHECK, GenJob,
                                         check_contract, contract_rules,
                                         exemplar_topics, exemplars_block,
                                         keep_first_code_block, strip_imports,
                                         title_of)
from src.generation.plan_guides import _VIOLATION_HINTS
from src.generation.rag_guides import _FENCE_RE, _TITLE_RE, _code_ok, _slug
from src.utils import DOCUMENT_SRC_DOCUMENTS
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)

_INTENT_SPLIT = re.compile(r"^\s*#{0,3}\s*INTENT\s*\d*\s*#*\s*$", re.M | re.I)
_A_MARKER = "\nA:\n"
_SOLSTART_MARKER = "# SOLUTION START"


def _condition_text(prompt: Text) -> Text:
    """The task's condition only: everything before the harness scaffold.

    Never reads ``reference_code`` — the caller does not even pass it in.
    """
    prompt = prompt or ""
    idx = prompt.find(_A_MARKER)
    if idx == -1:
        idx = prompt.find(_SOLSTART_MARKER)
    return (prompt[:idx] if idx != -1 else prompt).strip()


def _indent(code: str) -> str:
    return "\n".join("    " + l for l in code.rstrip().split("\n")) + "\n"


class IntentHowtoGenerator(PromptDiscoverable, Generator):
    """DS1000 condition -> intent -> howto document (semi-oracle).

    Args:
        url: base URL of the OpenAI-compatible LLM endpoint (stages 1 and 3).
        model_name: model id to call; resolved from the endpoint if unset.
        embedder: embedding agent; required when ``clusterize`` is True.
        dataset_path: path to the DS1000 ``.jsonl.gz`` (source of conditions).
        intents_cache_path: where the mined intents table is cached (jsonl).
        rebuild_cache: force re-mining stage 1 even if the cache exists.
        n_intents_per_task: purposes mined per DS1000 condition (default 1).
        dedup_by_origin: mine only one task per ``perturbation_origin_id``
            (the lowest ``problem_id`` in the group), since DS1000's
            Surface/Semantic/Difficult-Rewrite variants restate the same
            underlying problem.
        clusterize: group intents by embedding similarity and write one
            document per cluster representative instead of per intent.
        n_clusters: number of clusters; 0 = auto (``round(sqrt(n))``).
        pca_n_components: dimensionality PCA reduces to before clustering.
        fewshot_db_path / n_fewshot: manual corpus used as FORM exemplars for
            stage 3 (see ``src.generation.doc_contract.exemplars_block``).
        num_workers: parallel LLM calls per stage.
        temperature_intent: sampling temperature for stage 1.
        temperature_doc: sampling temperature for stage 3.
        max_condition_chars: chars of the condition shown to the LLM in stage 1.
        max_doc_chars: drop generated docs longer than this (chunker budget).
        max_retries: corrective retries per document when the contract rejects.
        embed_batch_size: intents per embedder call in stage 2.
        seed: base seed for reproducible sampling/clustering/LLM calls.
        limit: if > 0, only mine intents for the first ``limit`` DS1000 tasks
            (smoke runs). 0 = all.
        dump_path: if set, append surviving docs to this jsonl file (audit).
        cluster_dump_path: if set, dump cluster membership to this jsonl file.
        reject_dump_path: rejected drafts, so a shrinking yield is auditable.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        embedder: Agent = None,
        dataset_path: str = "data/ds1000/ds1000.jsonl.gz",
        intents_cache_path: str = "data/intent_howto_intents.jsonl",
        rebuild_cache: bool = False,
        n_intents_per_task: int = 1,
        dedup_by_origin: bool = True,
        clusterize: bool = False,
        n_clusters: int = 0,
        pca_n_components: int = 10,
        fewshot_db_path: str = "data/docs_database_examples_aug.db",
        n_fewshot: int = 3,
        num_workers: int = 8,
        temperature_intent: float = 0.7,
        temperature_doc: float = 0.3,
        max_condition_chars: int = 1200,
        max_doc_chars: int = 1100,
        max_retries: int = 2,
        embed_batch_size: int = 32,
        seed: int = 0,
        limit: int = 0,
        dump_path: str = "",
        cluster_dump_path: str = "",
        reject_dump_path: str = "",
        name: str = "intent_howto_generator",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.embedder = embedder
        self.dataset_path = dataset_path
        self.intents_cache_path = intents_cache_path
        self.rebuild_cache = bool(rebuild_cache)
        self.n_intents_per_task = max(1, int(n_intents_per_task))
        self.dedup_by_origin = bool(dedup_by_origin)
        self.clusterize = bool(clusterize)
        self.n_clusters = int(n_clusters)
        self.pca_n_components = max(2, int(pca_n_components))
        self.fewshot_db_path = fewshot_db_path
        self.n_fewshot = int(n_fewshot)
        self.num_workers = max(1, int(num_workers))
        self.temperature_intent = float(temperature_intent)
        self.temperature_doc = float(temperature_doc)
        self.max_condition_chars = int(max_condition_chars)
        self.max_doc_chars = int(max_doc_chars)
        self.max_retries = max(0, int(max_retries))
        self.embed_batch_size = int(embed_batch_size)
        self.seed = int(seed)
        self.limit = int(limit)
        self.dump_path = dump_path
        self.cluster_dump_path = cluster_dump_path
        self.reject_dump_path = reject_dump_path
        self.stats: Counter = Counter()
        self.rejects: List[dict] = []
        self._seed_supported = True

    # -------------------------------------------------------------------- LLM
    def _model(self) -> str:
        if not self.model_name:
            self.model_name = self.client.models.list().data[0].id
            logger.info("%s: model resolved from endpoint -> %s",
                        self.name, self.model_name)
        return self.model_name

    def _chat(self, system: Text, user: Text, temperature: float,
              seed: int = 0) -> Text:
        """One chat call. If the endpoint rejects `seed`, drop it and say so.

        Same fallback as PlanGuideGenerator._complete — an unsupported `seed`
        kwarg must not silently empty the whole generation stage.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = dict(model=self._model(), messages=messages,
                      temperature=temperature)
        if self._seed_supported:
            kwargs["seed"] = seed
        try:
            response = self.client.chat.completions.create(**kwargs)
        except TypeError as exc:
            if not self._seed_supported or "seed" not in str(exc):
                raise
            self._seed_supported = False
            logger.warning("%s: endpoint rejected the seed parameter (%s) — "
                           "continuing WITHOUT per-call seeds, runs will not "
                           "be bit-reproducible", self.name, exc)
            kwargs.pop("seed", None)
            response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    # ------------------------------------------------------ stage 1: mining
    def _load_tasks(self) -> List[dict]:
        tasks = []
        with gzip.open(self.dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                tasks.append(json.loads(line))
        if self.dedup_by_origin:
            by_origin: Dict[object, dict] = {}
            for task in tasks:
                meta = task.get("metadata", {}) or {}
                origin = meta.get("perturbation_origin_id", meta.get("problem_id"))
                current = by_origin.get(origin)
                if current is None or meta.get("problem_id", 0) < \
                        (current.get("metadata", {}) or {}).get("problem_id", 0):
                    by_origin[origin] = task
            tasks = sorted(by_origin.values(),
                           key=lambda t: (t.get("metadata", {}) or {}).get("problem_id", 0))
            self.stats["tasks_after_origin_dedup"] = len(tasks)
        if self.limit and self.limit < len(tasks):
            tasks = tasks[: self.limit]
        return tasks

    def _intents_for_task(self, task: dict, task_index: int) -> List[Text]:
        meta = task.get("metadata", {}) or {}
        lib = meta.get("library") or "the library"
        condition = _condition_text(task.get("prompt", ""))[: self.max_condition_chars]

        system = self.prompt("intent_system", (
            "You are a senior Python engineer who knows why practitioners reach "
            "for a given operation."
        ))
        user = self.render_prompt("intent_main", (
            "Below is the CONDITION of a $lib programming task, taken from a "
            "benchmark (never its solution).\n-----\n$condition\n-----\n"
            "Write $n different one-line PURPOSE statements: the general, "
            "reusable question ('How do I ...?') a real developer would type "
            "into a search box when facing a problem LIKE this one. Generalize "
            "away from the specific numbers/example shown above — do not "
            "restate them. Do NOT write code, do NOT mention 'task' or "
            "'benchmark', and do NOT copy any harness wording (BEGIN SOLUTION, "
            "SOLUTION START, <code> tags, 'result = ...').\n"
        ), lib=lib, condition=condition, n=self.n_intents_per_task)
        tail = self.prompt("intent_tail", (
            "Format STRICTLY as:\n### INTENT 1\n<one line>\n"
            "### INTENT 2\n<one line>\n..."
        ), optimizable=False)

        text = self._chat(system, user + tail, self.temperature_intent,
                          seed=self.seed + task_index)
        return self._parse_intents(text)

    def _parse_intents(self, text: Text) -> List[Text]:
        parts = _INTENT_SPLIT.split(text or "")
        candidates = parts[1:] if len(parts) > 1 else parts
        intents = []
        for part in candidates:
            line = part.strip().splitlines()[0].strip() if part.strip() else ""
            line = line.lstrip("-•* ").strip()
            if line and line not in intents:
                intents.append(line)
        return intents[: self.n_intents_per_task]

    def _mine_intents_table(self, tasks: List[dict]) -> List[dict]:
        def work(task: dict, index: int):
            return task, self._intents_for_task(task, index)

        results = []
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as ex:
            futures = [ex.submit(work, t, i) for i, t in enumerate(tasks)]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{self.name}:mine-intents"):
                try:
                    results.append(fut.result())
                except Exception as exc:   # one bad task must not kill the run
                    logger.warning("intent mining failed for a task: %s", exc)

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
                })

        os.makedirs(os.path.dirname(self.intents_cache_path) or ".", exist_ok=True)
        with open(self.intents_cache_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("%s: mined %d intents from %d tasks -> %s",
                    self.name, len(rows), len(tasks), self.intents_cache_path)
        return rows

    def _load_or_build_intents_table(self) -> List[dict]:
        if not self.rebuild_cache and os.path.isfile(self.intents_cache_path):
            rows = []
            with open(self.intents_cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            logger.info("%s: loaded %d cached intents from %s",
                        self.name, len(rows), self.intents_cache_path)
            return rows
        return self._mine_intents_table(self._load_tasks())

    # --------------------------------------------------- stage 2: clustering
    def _embed_all(self, texts: List[Text]) -> np.ndarray:
        out: List[np.ndarray] = []
        bs = self.embed_batch_size
        for start in tqdm(range(0, len(texts), bs), desc=f"{self.name}:embed"):
            vectors = self.embedder.run(texts[start:start + bs])
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

    def _groups(self, rows: List[dict]) -> List[Tuple[int, List[int]]]:
        """Return [(representative_index, member_indices), ...] over ``rows``.

        Without ``clusterize`` every intent is its own singleton group.
        """
        n = len(rows)
        if not self.clusterize or n <= 2:
            if self.clusterize and n <= 2:
                logger.info("%s: only %d intents — too few to cluster, "
                            "treating each as its own group", self.name, n)
            return [(i, [i]) for i in range(n)]
        if self.embedder is None:
            raise ValueError(f"{self.name}: clusterize=True needs an embedder "
                             "wired into the pipeline")

        vecs = self._normalize(self._embed_all([r["intent"] for r in rows]))

        k = self.n_clusters if self.n_clusters > 0 else max(2, round(n ** 0.5))
        k = max(1, min(k, n))

        # Curse of dimensionality: high-dimensional embedding vectors have
        # pairwise distances that concentrate, so raw k-means would cluster
        # on noise rather than semantics. PCA projects onto the low-
        # dimensional subspace the embeddings actually vary along first —
        # same purpose as RAPTOR's UMAP step, but pure numpy/scipy (umap-learn
        # pulls in numba, which needs numpy<=2.4 and conflicts with this
        # project's numpy 2.5).
        n_components = min(self.pca_n_components, n - 1, vecs.shape[1])
        if n_components >= 2:
            reduced = PCA(n_components=n_components,
                          random_state=self.seed).fit_transform(vecs)
        else:
            reduced = vecs

        labels = KMeans(n_clusters=k, n_init=10,
                        random_state=self.seed).fit_predict(reduced)
        self.stats["clusters"] = len(set(labels.tolist()))
        self.stats["intents_total"] = n

        by_label: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            by_label[int(label)].append(i)

        groups: List[Tuple[int, List[int]]] = []
        for idxs in by_label.values():
            if len(idxs) == 1:
                groups.append((idxs[0], idxs))
                continue
            # Medoid in the ORIGINAL normalized embedding space: UMAP's
            # coordinates are a clustering aid, not a similarity measure to
            # trust for picking the intent that speaks for the group.
            sub = vecs[idxs]
            centroid = sub.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid) or 1.0
            sims = sub @ (centroid / centroid_norm)
            representative = idxs[int(np.argmax(sims))]
            groups.append((representative, idxs))
        return groups

    # ----------------------------------------------------- stage 3: writing
    def _howto_job(self, row: dict, rotation: int, plan_id: str) -> GenJob:
        prompt = self.render_prompt(
            "howto", _HOWTO_PROMPT,
            lib=row["library"] or "Python",
            intent=row["intent"],
            rules=contract_rules(DOC_CLASS_RECIPE),
            examples=exemplars_block(self.fewshot_db_path, DOC_CLASS_RECIPE,
                                     rotation=rotation, n=self.n_fewshot),
            selfcheck=SELF_CHECK)
        return GenJob(library=row["library"] or "", doc_class=DOC_CLASS_RECIPE,
                      prompt=prompt, plan_id=plan_id, topic=row["intent"])

    def _parse_doc(self, text: str) -> Optional[str]:
        text = (text or "").strip()
        if not text:
            return None
        m = _TITLE_RE.search(text)
        title = m.group(1).strip() if m else text.splitlines()[0][:80]
        body = _TITLE_RE.sub("", text, count=1).strip()
        content = title + "\n\n" + body
        blocks = _FENCE_RE.findall(content)
        if any(not _code_ok(b) for b in blocks):
            self.stats["drop_bad_code"] += 1
            return None
        content = _FENCE_RE.sub(lambda mm: _indent(mm.group(1)), content).strip() + "\n"
        if not (120 <= len(content) <= self.max_doc_chars):
            self.stats["drop_length"] += 1
            return None
        return content

    def _write_doc(self, job: GenJob, index: int
                   ) -> Optional[Tuple[GenJob, str]]:
        ex_topics = exemplar_topics(self.fewshot_db_path, job.doc_class)
        note = ""
        for attempt in range(1, self.max_retries + 2):
            try:
                raw = self._chat(
                    self.prompt("chat_system", (
                        "You are a senior Python engineer writing short, "
                        "precise knowledge-base documents for developers who "
                        "complete partially written code snippets."
                    )),
                    job.prompt + note, self.temperature_doc,
                    seed=self.seed + index * 17 + attempt)
            except Exception as exc:
                self.stats["drop_call_error"] += 1
                logger.warning("generation call failed (%s): %s", job.plan_id, exc)
                return None
            content = self._parse_doc(raw)
            if content is None:
                self.stats["drop_parse"] += 1
                note = "\nYour previous answer was not parseable. Follow the "\
                       "TITLE: format exactly."
                continue
            content, n_imports = strip_imports(content)
            content, n_blocks = keep_first_code_block(content)
            self.stats["repaired_imports"] += n_imports
            self.stats["repaired_extra_blocks"] += n_blocks
            violation, soft = check_contract(content, job.doc_class,
                                             job.api_token, job.angle, ex_topics)
            if violation is None:
                self.stats["parsed_ok"] += 1
                return job, content
            self.stats[f"reject_{violation}"] += 1
            self.rejects.append({"plan_id": job.plan_id, "library": job.library,
                                 "attempt": attempt, "violation": violation,
                                 "content": content})
            note = "\nYour previous answer was rejected: " + \
                _VIOLATION_HINTS.get(violation.split(":")[0], violation) + \
                " Rewrite it following the requirements exactly."
        self.stats["retry_exhausted"] += 1
        return None

    def _run_jobs(self, jobs: List[GenJob]) -> List[Tuple[GenJob, str]]:
        results: List[Optional[Tuple[GenJob, str]]] = [None] * len(jobs)
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as ex:
            futures = {ex.submit(self._write_doc, job, i): i
                       for i, job in enumerate(jobs)}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{self.name}:write"):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:      # one bad job must not kill the run
                    self.stats["drop_job_error"] += 1
                    logger.warning("job %d failed: %s", i, exc)
        return [r for r in results if r is not None]

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        self.stats = Counter()
        self.rejects = []

        rows = self._load_or_build_intents_table()
        if not rows:
            logger.warning("%s: no intents mined/loaded, nothing to do", self.name)
            return []

        groups = self._groups(rows)
        self.stats["groups"] = len(groups)
        logger.info("%s: %d intents -> %d group(s) to write (clusterize=%s)",
                    self.name, len(rows), len(groups), self.clusterize)

        if self.cluster_dump_path:
            os.makedirs(os.path.dirname(self.cluster_dump_path) or ".", exist_ok=True)
            with open(self.cluster_dump_path, "w", encoding="utf-8") as f:
                for cluster_id, (rep, members) in enumerate(groups):
                    f.write(json.dumps({
                        "cluster_id": cluster_id,
                        "representative": rows[rep]["intent"],
                        "members": [rows[m]["intent"] for m in members],
                    }, ensure_ascii=False) + "\n")

        jobs = []
        for cluster_id, (rep, members) in enumerate(groups):
            row = rows[rep]
            plan_id = f"cluster{cluster_id}:{row.get('problem_id')}:{row.get('order_id')}"
            jobs.append(self._howto_job(row, rotation=cluster_id, plan_id=plan_id))
        if self.limit:
            jobs = jobs[: self.limit]

        produced = self._run_jobs(jobs)
        return self._package(produced, documents, rows, groups)

    def _package(self, produced: List[Tuple[GenJob, str]],
                 documents: List[Document], rows: List[dict],
                 groups: List[Tuple[int, List[int]]]) -> List[Document]:
        cluster_size = {f"cluster{i}:{rows[rep].get('problem_id')}:"
                        f"{rows[rep].get('order_id')}": len(members)
                        for i, (rep, members) in enumerate(groups)}
        ids_offset = len(documents) + 1
        seen_names = set()
        synth: List[Document] = []
        dump_f = None
        if self.dump_path:
            os.makedirs(os.path.dirname(self.dump_path) or ".", exist_ok=True)
            dump_f = open(self.dump_path, "w", encoding="utf-8")
        try:
            for job, content in produced:
                doc_name = _slug(title_of(content), "howto")
                if doc_name in seen_names:
                    self.stats["drop_dup_name"] += 1
                    continue
                seen_names.add(doc_name)
                synth.append(Document(
                    id=str(len(synth) + ids_offset),
                    text=content,
                    source=DOCUMENT_SRC_DOCUMENTS,
                    metadata={
                        "library": job.library,
                        "section": "intent_howto_gen",
                        "doc_name": doc_name,
                        "plan_id": job.plan_id,
                        "intent": job.topic,
                        "cluster_size": cluster_size.get(job.plan_id, 1),
                    },
                ))
                if dump_f:
                    dump_f.write(json.dumps({
                        "library": job.library, "doc_name": doc_name,
                        "plan_id": job.plan_id, "intent": job.topic,
                        "cluster_size": cluster_size.get(job.plan_id, 1),
                        "content": content,
                    }, ensure_ascii=False) + "\n")
        finally:
            if dump_f:
                dump_f.close()

        if self.reject_dump_path and self.rejects:
            os.makedirs(os.path.dirname(self.reject_dump_path) or ".", exist_ok=True)
            with open(self.reject_dump_path, "w", encoding="utf-8") as f:
                for row in self.rejects:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(
            "%s: %d documents into the corpus (from %d group(s), clusterize=%s). "
            "Yield stats: %s", self.name, len(synth), len(groups),
            self.clusterize, dict(self.stats))
        return synth


_HOWTO_PROMPT = """\
You are writing one document for a knowledge base that is read by a developer
who is completing a partially written $lib snippet. The imports and the data
are already in place above the gap; the developer needs the missing lines and
nothing else.

Purpose of this document: answer, in one short recipe, the practical question
below the way an experienced $lib user would.

The question it answers: $intent

Form:
$rules- under 120 words in total.

$examples
$selfcheck
Answer STRICTLY as:
TITLE: <the question, phrased naturally>
<the document>
"""
