"""PlanGuideGenerator — plan-first generation of RAG instruction documents (exp17).

What the exp15 post-mortem established (Progress.md, S1-S8):

* generated documents ARE retrieved — more often than the manual ones — so the
  bottleneck is content, not delivery;
* the exp14/15 generators wrote about the most frequent APIs (median corpus
  frequency 398) while the manual documents that moved the metric write about
  specific ones (median 31);
* one template produced near-duplicates that then dominated retrieval: a single
  document sat in 205 of 1000 contexts and did nothing;
* every generated example was a standalone script with imports and invented
  data, i.e. the opposite of the completion protocol the reader needs.

This generator answers those four points:

1. **corpus-grounded signals.** Topics are mined from the corpus itself — rare
   APIs, signature defaults, deprecation sentences, look-alike API pairs — never
   invented by the model from a blank prompt (that is what produced "how do I
   create a simple line plot" in exp14).
2. **a plan before the documents.** One planning pass turns the evidence into a
   typed list of documents (class / library / trap / api / vars) in JSONL; every
   subsequent planning call is shown the titles already accepted, so the model
   does not propose them again. The plan is deduplicated lexically and, when an
   embedder is wired in, semantically — BEFORE any document is written.
3. **plan execution control.** Each plan item is tracked to a terminal status
   (generated / duplicate / rejected:<rule> / parse_failed / retry_exhausted) and
   the coverage report goes to the log and into the materialized .db, so a
   silently shrinking yield is visible instead of being discovered months later.
4. **the shared document contract** (`src.generation.doc_contract`), identical to
   the one the repaired ApiGuideGenerator uses, so a comparison between the two
   methods is a comparison of topic selection only.

Determinism: jobs are ordered, sets are sorted before use, and every LLM call
gets its own ``seed = base_seed + job_index`` so batching does not change the
result. Full reproducibility comes from the materialized corpus .db, not from
the seed alone.

NB: no ``from __future__ import annotations`` — the component registry
introspects real annotation objects to resolve dependencies.
"""

import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Document
from src.agent_constructor.generator import Generator
from src.agent_constructor.prompts import PromptDiscoverable
from src.generation.api_guides import api_census, rare_api_candidates
from src.generation.doc_contract import (DOC_CLASS_PROTOCOL, DOC_CLASS_RECIPE,
                                         GenJob, check_contract, contract_rules,
                                         protocol_jobs, title_of)
from src.generation.rag_guides import _FENCE_RE, _TITLE_RE, _code_ok, _slug
from src.utils import DOCUMENT_SRC_DOCUMENTS
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)

# A deprecation sentence is only useful if it names something. Cutting at the
# first period produced "Changed in version 1." — version numbers are periods
# too — so the window is bounded by the line and then filtered on content.
_DEPRECATION_RE = re.compile(
    r"[^\n]{0,130}?(?:deprecated since|will be removed|has been removed|"
    r"is removed|was renamed|renamed to|use\s+\S+\s+instead)[^\n]{0,150}", re.I)
_NAMED_THING_RE = re.compile(r"[A-Za-z_]\w*[._]\w+|`[^`]+`")
_SIGNATURE_RE = re.compile(
    r"\b([A-Za-z_][\w.]*)\s*\(\s*([^)]{0,200}?=\s*[^)]{0,200})\)")
# Keywords whose default silently changes the RESULT (not the dtype or a copy):
# these are what the manual recipes disambiguate.
_INTERESTING_KWARG_RE = re.compile(
    r"\b(axis|inplace|keepdims|numeric_only|dropna|ddof|ascending|normalize|"
    r"bias|closed|origin|drop_first|as_index|sort_index|na_position)\s*=", re.I)
_PRIVATE_RE = re.compile(r"(?:^|\.)_")
_WORD_RE = re.compile(r"[a-z_]{3,}")


def _is_public(api: str) -> bool:
    """Reject dunders and private helpers — nobody writes .__finalize__()."""
    name = api.lstrip(".")
    return bool(name) and not name.startswith("_") and not _PRIVATE_RE.search(api[1:])


def _norm_topic(text: str) -> str:
    return " ".join(sorted(set(_WORD_RE.findall(text.lower()))))


def _cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


class PlanGuideGenerator(PromptDiscoverable, Generator):
    """Plan a set of documents from corpus evidence, then write them.

    Args:
        url: OpenAI-compatible endpoint of the writing model.
        model_name: model id.
        embedder: optional embedding agent; enables semantic dedup of the plan.
        path_to_db: sqlite path — its ``examples`` table joins the API census.
        n_docs: total document budget (protocol pack included).
        n_protocol_docs: how much of that budget the protocol class takes.
        plan: False runs the ablation — same signals, one document per signal,
            no planning pass and therefore no cross-item dedup.
        rare_api_min / rare_api_max: corpus-frequency window a recipe API must
            fall into (the manual recipes sit at median 31).
        num_workers: parallel LLM calls.
        temperature: sampling temperature for document writing.
        plan_temperature: sampling temperature for the planning pass.
        seed: base seed; call *i* uses ``seed + i``.
        max_retries: corrective retries per document when the contract rejects.
        max_evidence_per_lib: cap on mined evidence strings handed to the
            planner per library (the planner stops once the quota is met, so a
            bigger pool only costs memory).
        dedup_threshold: cosine above which two plan items count as the same.
        limit: cap on documents (smoke runs). 0 = no cap.
        max_doc_chars: drop documents longer than this (chunker budget).
        dump_path / plan_dump_path: audit jsonl files.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        embedder: Agent = None,
        path_to_db: str = "",
        n_docs: int = 150,
        n_protocol_docs: int = 24,
        plan: bool = True,
        rare_api_min: int = 5,
        rare_api_max: int = 100,
        num_workers: int = 8,
        temperature: float = 0.2,
        plan_temperature: float = 0.8,
        seed: int = 0,
        max_retries: int = 2,
        max_evidence_per_lib: int = 240,
        dedup_threshold: float = 0.85,
        limit: int = 0,
        max_doc_chars: int = 1100,
        dump_path: str = "",
        plan_dump_path: str = "",
        name: str = "plan_guide_generator",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.endpoint_url = url or ""
        self.model_name = model_name
        self.embedder = embedder
        self.path_to_db = path_to_db
        self.n_docs = int(n_docs)
        self.n_protocol_docs = int(n_protocol_docs)
        self.plan_enabled = bool(plan)
        self.rare_api_min = int(rare_api_min)
        self.rare_api_max = int(rare_api_max)
        self.num_workers = max(1, int(num_workers))
        self.temperature = float(temperature)
        self.plan_temperature = float(plan_temperature)
        self.seed = int(seed)
        self.max_retries = max(0, int(max_retries))
        self.max_evidence_per_lib = max(12, int(max_evidence_per_lib))
        self.dedup_threshold = float(dedup_threshold)
        self.limit = int(limit)
        self.max_doc_chars = int(max_doc_chars)
        self.dump_path = dump_path
        self.plan_dump_path = plan_dump_path

        self._seed_supported = True
        self.stats: Counter = Counter()
        self.audit_docs: Dict[str, dict] = {}
        self.audit_plans: List[dict] = []

    # -------------------------------------------------------------------- LLM

    def _complete(self, messages, temperature: float, seed: int) -> str:
        """One chat call. If the endpoint rejects `seed`, drop it and say so.

        Losing reproducibility is bad; losing the whole generation stage to an
        unsupported parameter is worse, and it would look like an empty corpus
        rather than an error (see the two silent no-ops in Progress.md).
        """
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
                           "continuing WITHOUT per-call seeds, runs will not be "
                           "bit-reproducible", self.name, exc)
            kwargs.pop("seed", None)
            response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _model(self) -> str:
        """Model id, resolved from the endpoint when the config does not pin it.

        DS1000Solver already works this way; pinning the id in yaml only creates
        a second place to keep in sync with whatever vLLM currently serves.
        """
        if not self.model_name:
            self.model_name = self.client.models.list().data[0].id
            logger.info("%s: model resolved from endpoint -> %s",
                        self.name, self.model_name)
        return self.model_name
    def _chat(self, user: str, seed: Optional[int] = None,
              temperature: Optional[float] = None) -> str:
        messages = [
            {"role": "system",
             "content": self.prompt("chat_system", (
                 "You are a senior Python engineer writing short, precise "
                 "knowledge-base documents for developers who complete "
                 "partially written code snippets."
             ))},
            {"role": "user", "content": user},
        ]
        return self._complete(
            messages,
            self.temperature if temperature is None else temperature,
            self.seed if seed is None else seed)

    # ---------------------------------------------------------------- signals
    def _signals(self, documents: List[Document]) -> Dict[str, List[str]]:
        """Mine per-library evidence strings from the corpus itself.

        Evidence of the four kinds is interleaved rather than concatenated: the
        planner only ever sees the first few batches (it stops once the library
        quota is planned), and a sorted list would hand it rare APIs only, never
        a deprecation or a look-alike pair.
        """
        by_kind: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list))

        # (1) rare-but-recurring APIs — the specificity band the manual recipes
        # occupy; api_census is the repaired census from api_guides.
        census = api_census(documents, self.path_to_db, self.stats)
        rare = [(lib, api, count) for lib, api, count
                in rare_api_candidates(census, self.rare_api_min, self.rare_api_max)
                if _is_public(api)]
        for lib, api, count in rare:
            by_kind[lib]["api"].append(
                f"API {api} is used {count} times in the corpus")
        self.stats["signal_rare_api"] = len(rare)

        # (2) signature defaults, (3) deprecation sentences, (4) look-alike pairs
        names_by_lib: Dict[str, set] = defaultdict(set)
        for doc in documents:
            lib = (doc.metadata or {}).get("library")
            if not lib:
                continue
            text = doc.text or ""
            for m in _SIGNATURE_RE.finditer(text[:4000]):
                func = m.group(1)
                if not _is_public(func) or not _INTERESTING_KWARG_RE.search(m.group(2)):
                    continue
                sig = re.sub(r"\s+", " ", m.group(0))[:160]
                by_kind[lib]["default"].append(f"signature default: {sig}")
                names_by_lib[lib].add(func.rsplit(".", 1)[-1])
                break
            for m in _DEPRECATION_RE.finditer(text):
                note = re.sub(r"\s+", " ", m.group(0)).strip()
                # Keep only sentences that actually name an API; "Changed in
                # version 2." is true and useless.
                if len(note) >= 45 and _NAMED_THING_RE.search(note):
                    by_kind[lib]["deprecation"].append("deprecation note: " + note[:200])
                    break

        # Rank look-alike pairs by how much the corpus uses them, not
        # alphabetically — sorted order gave acos/acosh and never sort/argsort.
        freq = {}
        for (lib, api), count in census.items():
            freq.setdefault(lib, {})
            name = api.rsplit(".", 1)[-1].lstrip(".")
            freq[lib][name] = max(freq[lib].get(name, 0), count)
        for lib, names in names_by_lib.items():
            known = sorted(n for n in names
                           if _is_public(n) and freq.get(lib, {}).get(n, 0) >= self.rare_api_min)
            pairs = _lookalike_pairs(known)
            pairs.sort(key=lambda pr: (-min(freq[lib].get(pr[0], 0),
                                            freq[lib].get(pr[1], 0)), pr))
            for one, two in pairs[:60]:
                by_kind[lib]["pair"].append(
                    f"look-alike pair: {one} ({freq[lib].get(one, 0)} uses) and "
                    f"{two} ({freq[lib].get(two, 0)} uses) are easy to mix up")

        by_lib: Dict[str, List[str]] = {}
        for lib, kinds in by_kind.items():
            # Dedup while PRESERVING order: each kind was built in a meaningful
            # one (APIs by descending corpus use, pairs by descending use of the
            # rarer half). Re-sorting the strings alphabetically here used to
            # hand the planner ".abspath" instead of "np.zeros_like".
            ordered = {k: list(dict.fromkeys(v)) for k, v in kinds.items()}
            by_lib[lib] = _interleave([ordered[k] for k in sorted(ordered)],
                                      self.max_evidence_per_lib)
            for kind, values in sorted(ordered.items()):
                self.stats[f"signal_{kind}_{lib}"] = len(values)
            self.stats[f"signals_{lib}"] = len(by_lib[lib])
        return by_lib

    # ------------------------------------------------------------------- plan
    def _plan(self, signals: Dict[str, List[str]], quotas: Dict[str, int]
              ) -> List[dict]:
        """Planning pass: evidence -> typed plan items, deduplicated."""
        items: List[dict] = []
        accepted_titles: List[str] = []
        batch_size = 12
        call = 0
        for lib in sorted(signals):
            quota = quotas.get(lib, 0)
            if quota <= 0:
                continue
            evidence = signals[lib]
            per_call = max(4, min(12, quota))
            for start in range(0, len(evidence), batch_size):
                if sum(1 for i in items if i["lib"] == lib) >= quota:
                    break
                chunk = evidence[start:start + batch_size]
                prompt = self.render_prompt(
                    "plan", _PLAN_PROMPT,
                    lib=lib, n=per_call,
                    evidence="\n".join("- " + e for e in chunk),
                    avoid=("\n".join("- " + t for t in accepted_titles[-40:])
                           or "- (nothing yet)"))
                call += 1
                try:
                    raw = self._chat(prompt, seed=self.seed + 100000 + call,
                                     temperature=self.plan_temperature)
                except Exception as exc:
                    self.stats["plan_call_error"] += 1
                    logger.warning("plan call failed for %s: %s", lib, exc)
                    continue
                parsed = _parse_plan(raw, lib)
                self.stats["plan_items_raw"] += len(parsed)
                self.audit_plans.append(
                    {"library": lib, "status": "raw_plan", "plan_id": f"call{call}",
                     "topic": "", "raw": raw[:8000]})
                for item in parsed:
                    items.append(item)
                    accepted_titles.append(item["topic"])
        return items

    def _dedup_plan(self, items: List[dict], corpus_titles: set) -> List[dict]:
        """Lexical dedup, then embedding dedup when an embedder is available."""
        kept: List[dict] = []
        seen_keys = set()
        for item in items:
            key = _norm_topic(item["topic"])
            if not key:
                self.stats["plan_drop_empty"] += 1
                self._record_plan(item, "rejected:empty_topic")
                continue
            if key in seen_keys:
                self.stats["plan_drop_dup_lexical"] += 1
                self._record_plan(item, "duplicate")
                continue
            if key in corpus_titles:
                self.stats["plan_drop_covered_by_corpus"] += 1
                self._record_plan(item, "duplicate:corpus")
                continue
            seen_keys.add(key)
            kept.append(item)

        if self.embedder is None or len(kept) < 2:
            if self.embedder is None:
                logger.warning("no embedder wired into %s — plan dedup is "
                               "lexical only", self.name)
            return kept
        try:
            vectors = self.embedder.run([i["topic"] for i in kept])
        except Exception as exc:
            logger.warning("plan embedding failed (%s) — lexical dedup only", exc)
            return kept
        final: List[dict] = []
        chosen: List[List[float]] = []
        for item, vec in zip(kept, vectors):
            if any(_cosine(vec, other) > self.dedup_threshold for other in chosen):
                self.stats["plan_drop_dup_semantic"] += 1
                self._record_plan(item, "duplicate:semantic")
                continue
            chosen.append(vec)
            final.append(item)
        return final

    def _record_plan(self, item: dict, status: str) -> None:
        self.audit_plans.append({
            "library": item.get("lib", ""), "status": status,
            "plan_id": item.get("plan_id", ""), "topic": item.get("topic", ""),
            "raw": json.dumps(item, ensure_ascii=False)})

    # ------------------------------------------------------------------- jobs
    def _recipe_job(self, item: dict) -> GenJob:
        api = item.get("api", "")
        prompt = self.render_prompt(
            "recipe", _RECIPE_PROMPT,
            lib=item["lib"], topic=item["topic"], trap=item.get("trap", ""),
            api=api or "the relevant API",
            vars=item.get("vars", ", ".join(("a", "arr", "df", "x"))),
            rules=contract_rules(DOC_CLASS_RECIPE, api))
        return GenJob(library=item["lib"], doc_class=DOC_CLASS_RECIPE,
                      prompt=prompt, api_token=api, plan_id=item["plan_id"],
                      topic=item["topic"])

    def _jobs_without_plan(self, signals: Dict[str, List[str]],
                           quotas: Dict[str, int]) -> List[GenJob]:
        """Ablation: one document per signal, no planning pass, no plan dedup."""
        jobs: List[GenJob] = []
        for lib in sorted(signals):
            quota = quotas.get(lib, 0)
            for i, evidence in enumerate(signals[lib][:quota]):
                api = ""
                m = re.match(r"API (\S+) ", evidence)
                if m:
                    api = m.group(1)
                item = {"lib": lib, "topic": evidence, "trap": evidence,
                        "api": api, "vars": "a, arr, df, x",
                        "plan_id": f"noplan:{lib}:{i}"}
                jobs.append(self._recipe_job(item))
        return jobs

    # ------------------------------------------------------------- production
    def _write_doc(self, job: GenJob, index: int) -> Optional[Tuple[GenJob, str, int]]:
        """Generate one document, validate it, retry once on a contract miss."""
        note = ""
        for attempt in range(1, self.max_retries + 2):
            try:
                raw = self._chat(job.prompt + note, seed=self.seed + index * 17 + attempt)
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
            violation, soft = check_contract(
                content, job.doc_class, job.api_token, job.require_facts)
            if violation is None:
                self.stats["parsed_ok"] += 1
                for flag, value in soft.items():
                    self.stats[f"soft_{flag}"] += int(bool(value))
                return job, content, attempt
            self.stats[f"reject_{violation}"] += 1
            note = "\nYour previous answer was rejected: " + \
                _VIOLATION_HINTS.get(violation, violation) + \
                " Rewrite it following the requirements exactly."
        self.stats["retry_exhausted"] += 1
        self._fail(job, "retry_exhausted")
        return None

    def _fail(self, job: GenJob, status: str) -> None:
        self.audit_plans.append({
            "library": job.library, "status": status, "plan_id": job.plan_id,
            "topic": job.topic, "raw": ""})

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
        content = _FENCE_RE.sub(
            lambda mm: _indent(mm.group(1)), content).strip() + "\n"
        if not (120 <= len(content) <= self.max_doc_chars):
            self.stats["drop_length"] += 1
            return None
        return content

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        self.stats = Counter()
        self.audit_docs = {}
        self.audit_plans = []

        lib_counts = Counter((d.metadata or {}).get("library") for d in documents)
        libs = sorted(l for l in lib_counts if l)
        if not libs:
            logger.warning("%s: corpus has no library metadata", self.name)
            return []

        proto = protocol_jobs(libs, self.n_protocol_docs)
        recipe_budget = max(0, self.n_docs - len(proto))
        total = sum(lib_counts[l] for l in libs)
        quotas = {l: max(2, round(recipe_budget * lib_counts[l] / total))
                  for l in libs}

        signals = self._signals(documents)
        if self.plan_enabled:
            corpus_titles = {_norm_topic(title_of(d.text or "")) for d in documents}
            items = self._plan(signals, quotas)
            items = self._dedup_plan(items, corpus_titles)
            self.stats["plan_items_kept"] = len(items)
            recipe_jobs = [self._recipe_job(i) for i in items[:recipe_budget]]
        else:
            recipe_jobs = self._jobs_without_plan(signals, quotas)[:recipe_budget]
            self.stats["plan_items_kept"] = 0

        jobs = proto + recipe_jobs
        if self.limit:
            jobs = jobs[: self.limit]
        self.stats["jobs"] = len(jobs)
        logger.info("%s: %d jobs (%d protocol, %d recipe; plan=%s, seed=%d)",
                    self.name, len(jobs), len(proto), len(jobs) - len(proto),
                    self.plan_enabled, self.seed)

        produced = self._run_jobs(jobs)
        return self._package(produced, documents)

    def _run_jobs(self, jobs: List[GenJob]) -> List[Tuple[GenJob, str, int]]:
        results: List[Optional[Tuple[GenJob, str, int]]] = [None] * len(jobs)
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as ex:
            futures = {ex.submit(self._write_doc, job, i): i
                       for i, job in enumerate(jobs)}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=self.name):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:      # one bad job must not kill the run
                    self.stats["drop_job_error"] += 1
                    logger.warning("job %d failed: %s", i, exc)
        # Index order, not completion order: doc ids must not depend on timing.
        return [r for r in results if r is not None]

    def _package(self, produced: List[Tuple[GenJob, str, int]],
                 documents: List[Document]) -> List[Document]:
        seen_names = set()
        seen_keys = set()
        synth: List[Document] = []
        dump_f = None
        if self.dump_path:
            os.makedirs(os.path.dirname(self.dump_path) or ".", exist_ok=True)
            dump_f = open(self.dump_path, "w", encoding="utf-8")
        try:
            for job, content, attempts in produced:
                doc_name = _slug(title_of(content), job.prefix)
                key = _norm_topic(title_of(content))
                if doc_name in seen_names or key in seen_keys:
                    self.stats["drop_dup_output"] += 1
                    self._fail(job, "duplicate:output")
                    continue
                seen_names.add(doc_name)
                seen_keys.add(key)
                self.audit_docs[doc_name] = {
                    "doc_class": job.doc_class, "plan_id": job.plan_id,
                    "topic": job.topic, "api_token": job.api_token,
                    "prompt_sha1": hashlib.sha1(
                        job.prompt.encode("utf-8")).hexdigest()[:12],
                    "attempts": attempts}
                self.audit_plans.append({
                    "library": job.library, "status": "generated",
                    "plan_id": job.plan_id, "topic": job.topic, "raw": ""})
                synth.append(Document(
                    id=f"plan_gen_{len(synth)}",
                    text=content,
                    source=DOCUMENT_SRC_DOCUMENTS,
                    metadata={"library": job.library,
                              "section": f"generated_{job.doc_class}",
                              "doc_name": doc_name}))
                if dump_f:
                    dump_f.write(json.dumps(
                        {"library": job.library, "doc_name": doc_name,
                         "stage": job.doc_class, "plan_id": job.plan_id,
                         "topic": job.topic, "api": job.api_token,
                         "attempts": attempts, "content": content},
                        ensure_ascii=False) + "\n")
        finally:
            if dump_f:
                dump_f.close()

        if self.plan_dump_path:
            os.makedirs(os.path.dirname(self.plan_dump_path) or ".", exist_ok=True)
            with open(self.plan_dump_path, "w", encoding="utf-8") as f:
                for row in self.audit_plans:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        by_class = Counter(d.metadata["section"] for d in synth)
        statuses = Counter(p["status"] for p in self.audit_plans)
        planned = self.stats.get("plan_items_kept", 0) or self.stats.get("jobs", 0)
        logger.info(
            "%s: %d documents into the corpus (%s). PLAN COVERAGE: planned %d, "
            "jobs %d, generated %d, lost %d. Statuses: %s. Yield stats: %s",
            self.name, len(synth), dict(by_class), planned,
            self.stats.get("jobs", 0), len(synth),
            max(0, self.stats.get("jobs", 0) - len(synth)),
            dict(statuses), dict(self.stats))
        return synth


def _indent(code: str) -> str:
    return "\n".join("    " + l for l in code.rstrip().split("\n")) + "\n"


def _interleave(groups: List[List[str]], cap: int) -> List[str]:
    """Round-robin over the evidence kinds so every batch carries a mix."""
    out: List[str] = []
    row = 0
    while len(out) < cap:
        added = False
        for group in groups:
            if row < len(group):
                out.append(group[row])
                added = True
                if len(out) >= cap:
                    break
        if not added:
            break
        row += 1
    return out


def _lookalike_pairs(names: List[str]) -> List[Tuple[str, str]]:
    """Names that share a stem — the cheap, corpus-derived confusion signal."""
    pairs = []
    for i, one in enumerate(names):
        for two in names[i + 1:]:
            if one == two or abs(len(one) - len(two)) > 6:
                continue
            if one.startswith(two) or two.startswith(one) or (
                    len(one) > 4 and len(two) > 4 and one[:4] == two[:4]):
                pairs.append((one, two))
    return pairs


def _parse_plan(raw: str, lib: str) -> List[dict]:
    """Parse the JSONL plan; tolerate fences, numbering and stray prose."""
    items = []
    for line in (raw or "").splitlines():
        line = line.strip().strip(",")
        if not line or line.startswith("```"):
            continue
        line = re.sub(r"^\s*[-*\d.)\s]+", "", line)
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        topic = str(data.get("topic") or data.get("title") or "").strip()
        if len(topic) < 10:
            continue
        api = str(data.get("api") or "").strip()
        items.append({
            "lib": str(data.get("lib") or lib).strip() or lib,
            "topic": topic,
            "trap": str(data.get("trap") or "").strip(),
            "api": api,
            "vars": str(data.get("vars") or "").strip(),
            "plan_id": f"{lib}:{_slug(topic, 'p')[:48]}",
        })
    return items


_VIOLATION_HINTS = {
    "import_in_recommended_code":
        "the recommended code contained an import — the reader's snippet has "
        "already imported everything.",
    "invented_data_in_recommended_code":
        "the recommended code built its own DataFrame/array — use the variables "
        "that already exist instead.",
    "protocol_title_is_question":
        "the title was a question — state the rule instead.",
    "recipe_title_not_question":
        "the title was not a question a user would type.",
    "protocol_code_too_long": "the recommended code was longer than 3 lines.",
    "recipe_code_too_long": "the recommended code was longer than 5 lines.",
    "code_too_long": "there was too much code overall.",
    "protocol_facts_missing":
        "the text did not state that the setup already ran, that the real "
        "inputs are hidden, or where the answer must be assigned.",
    "api_token_missing": "the requested API token never appeared in the text.",
    "harness_marker": "it mentioned benchmark harness markers.",
}

_PLAN_PROMPT = """\
You are planning a set of short knowledge-base documents about the Python \
library $lib, for developers who complete partially written code snippets.

Evidence collected from the library's own documentation and example code:
$evidence

Documents already planned (do NOT propose these again, or anything that would \
say the same thing in other words):
$avoid

Propose up to $n NEW documents. Each one must describe a concrete trap: a place \
where two similar APIs or two similar spellings are easy to mix up, a default \
argument that silently changes the result, or an API that was renamed or \
removed. Skip anything a competent developer already knows (creating a \
DataFrame, plotting a line, basic indexing) — those documents are worthless.

Answer with ONE JSON object per line and nothing else:
{"lib": "$lib", "topic": "<the user's question in one line>", "trap": "<what \
exactly gets confused, one line>", "api": "<the API token the document is \
about>", "vars": "<names of variables the reader already has, comma separated>"}
"""

_RECIPE_PROMPT = """\
Write ONE short knowledge-base document for $lib.

The question it answers: $topic
The trap it must resolve: $trap
The API it is about: $api
Variables the reader already has in scope: $vars

Requirements:
$rules- under 120 words plus the code.

Format STRICTLY as:
TITLE: <the question, containing $api>
<document text, code in ``` fences>
"""
