"""PlanGuideGenerator — plan-first generation of RAG instruction documents (exp18).

exp17 ran this generator and produced 86 documents that reached 2.1% of the
retrieved context; five of them were ever retrieved at all. The post-mortem
(Progress.md, exp17) found the cause in the topic selection, not in the
planning: topics were anchored on RARE APIs because the manual recipes have a
median API frequency of 31 — but that median is taken over a mixture whose 75th
percentile is 104 and whose maximum is 2770. The manual documents anchor on an
everyday operation and put the trap inside it. Selecting by rarity produced
tf.distribute internals and a document about scipy.special.ccdf, which does not
exist.

What this version does differently:

1. **topics are everyday operations with a trap.** Evidence is mined from the
   corpus as before, but the "API X is used N times" signal — 2119 of ~2800
   signals in exp17, carrying no trap at all — is gone. What remains: a frequent
   operation that has a look-alike neighbour, an argument default that changes
   the result, a deprecation. Measured before the run: 62% of the APIs selected
   this way occur in user task text against 21% for the exp17 rule.
2. **one code block, and it is the correct one.** exp17 asked for a WRONG block
   and got one in 90% of documents; two of those counter-examples were
   hallucinations shipped as runnable code.
3. **form is shown, not described.** Three documents from the manual corpus go
   into every prompt as FORM exemplars, rotated per job; a document that reuses
   an exemplar's subject is rejected.
4. **mechanical repair before rejection.** An import line or a second code block
   is stripped rather than costing the whole document — that failure mode ate
   23-56 jobs per exp17 config.
5. **a plan before the documents**, deduplicated lexically and (with an embedder
   wired in) semantically, with every plan item tracked to a terminal status and
   a coverage report in the log and in the materialized .db.

Determinism: jobs are ordered, sets are sorted before use, and every LLM call
gets its own ``seed = base_seed + job_index`` so batching does not change the
result. Full reproducibility comes from the materialized corpus .db.

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
from src.generation.api_guides import (api_census, head_api_candidates,
                                       is_public, lookalike_siblings)
from src.generation.doc_contract import (DOC_CLASS_PROTOCOL, DOC_CLASS_RECIPE,
                                         SELF_CHECK, GenJob, check_contract,
                                         contract_rules, exemplar_topics,
                                         exemplars_block, keep_first_code_block,
                                         protocol_jobs, strip_imports, title_of)
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
        head_api_min: a recipe API must be used at least this many times in the
            corpus. exp17 used the rare tail and only 21% of its APIs occurred
            in user task text; the head plus a confusability signal reaches 62%.
        fewshot_db_path / n_fewshot: manual corpus used as FORM exemplars.
        reject_dump_path: rejected drafts, so a shrinking yield is auditable.
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
        n_docs: int = 220,
        n_protocol_docs: int = 70,
        plan: bool = True,
        head_api_min: int = 100,
        fewshot_db_path: str = "data/docs_database_examples_aug.db",
        n_fewshot: int = 3,
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
        reject_dump_path: str = "",
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
        self.head_api_min = int(head_api_min)
        self.fewshot_db_path = fewshot_db_path
        self.n_fewshot = int(n_fewshot)
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
        self.reject_dump_path = reject_dump_path
        self.rejects: List[dict] = []

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

        # (1) everyday operations that have a look-alike neighbour. This
        # replaces the exp17 evidence "API X is used N times", which carried no
        # trap at all and made up 2119 of ~2800 signals — the planner saw almost
        # nothing else and duly wrote about tf.distribute internals.
        census = api_census(documents, self.path_to_db, self.stats)
        siblings = lookalike_siblings(census, self.head_api_min)
        head = head_api_candidates(census, self.head_api_min)
        n_ops = 0
        for lib, api, count in head:
            sibs = siblings.get((lib, api))
            if not sibs:
                continue
            names = ", ".join(f"{n} ({c} uses)" for n, c in sibs)
            by_kind[lib]["operation"].append(
                f"common operation {api} ({count} uses) sits next to {names} — "
                f"which one is right depends on what the user actually wants")
            n_ops += 1
        self.stats["signal_operation"] = n_ops

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
        for lib in sorted(set(list(names_by_lib) + list(freq))):
            names = set(names_by_lib.get(lib, set())) | set(freq.get(lib, {}))
            known = sorted(n for n in names
                           if is_public(n)
                           and freq.get(lib, {}).get(n, 0) >= self.head_api_min)
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
    def _recipe_job(self, item: dict, rotation: int = 0) -> GenJob:
        api = item.get("api", "")
        prompt = self.render_prompt(
            "recipe", _RECIPE_PROMPT,
            lib=item["lib"], topic=item["topic"], trap=item.get("trap", ""),
            api=api or "the relevant API",
            vars=item.get("vars") or ", ".join(("a", "arr", "df", "x")),
            rules=contract_rules(DOC_CLASS_RECIPE, api),
            examples=exemplars_block(self.fewshot_db_path, DOC_CLASS_RECIPE,
                                     rotation=rotation, n=self.n_fewshot),
            selfcheck=SELF_CHECK)
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
                m = re.match(r"common operation (\S+) ", evidence)
                if m:
                    api = m.group(1)
                item = {"lib": lib, "topic": evidence, "trap": evidence,
                        "api": api, "vars": "a, arr, df, x",
                        "plan_id": f"noplan:{lib}:{i}"}
                jobs.append(self._recipe_job(item, rotation=len(jobs)))
        return jobs

    # ------------------------------------------------------------- production
    def _write_doc(self, job: GenJob, index: int) -> Optional[Tuple[GenJob, str, int]]:
        """Generate one document, repair what is mechanical, then validate."""
        note = ""
        ex_topics = exemplar_topics(self.fewshot_db_path, job.doc_class)
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
            # Mechanical repair beats rejection: an import line and a second
            # code block are lint, not a reason to lose a finished document.
            content, n_imports = strip_imports(content)
            content, n_blocks = keep_first_code_block(content)
            self.stats["repaired_imports"] += n_imports
            self.stats["repaired_extra_blocks"] += n_blocks
            violation, soft = check_contract(
                content, job.doc_class, job.api_token, job.angle, ex_topics)
            if violation is None:
                self.stats["parsed_ok"] += 1
                for flag, value in soft.items():
                    self.stats[f"soft_{flag}"] += int(bool(value))
                return job, content, attempt
            self.stats[f"reject_{violation}"] += 1
            self.rejects.append({"plan_id": job.plan_id, "library": job.library,
                                 "doc_class": job.doc_class, "attempt": attempt,
                                 "violation": violation, "content": content})
            note = "\nYour previous answer was rejected: " + \
                _VIOLATION_HINTS.get(violation.split(":")[0], violation) + \
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

        proto = protocol_jobs(libs, self.n_protocol_docs,
                              render=self.render_prompt,
                              fewshot_db_path=self.fewshot_db_path,
                              n_fewshot=self.n_fewshot)
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
            recipe_jobs = [self._recipe_job(item, rotation=k)
                           for k, item in enumerate(items[:recipe_budget])]
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

        if self.reject_dump_path and self.rejects:
            os.makedirs(os.path.dirname(self.reject_dump_path) or ".", exist_ok=True)
            with open(self.reject_dump_path, "w", encoding="utf-8") as f:
                for row in self.rejects:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
    "import_present":
        "it contained an import — the reader's imports are already done.",
    "invented_data":
        "it built its own DataFrame/array — use the variables that already exist.",
    "more_than_one_code_block":
        "it had more than one code block. Write ONE block, the correct way, and "
        "say the warning in a sentence instead.",
    "code_too_long": "the code was longer than allowed.",
    "prose_too_long": "the text was longer than allowed.",
    "no_lexical_anchor":
        "it named nothing concrete — mention the variable and the API.",
    "protocol_title_is_question":
        "the title was a question — state the rule instead.",
    "recipe_title_not_question":
        "the title was not a question a user would type.",
    "protocol_fact_missing":
        "it did not state that the setup already ran, that the real inputs are "
        "hidden, or where the answer must be assigned.",
    "angle_not_stated": "it did not state the rule it was asked to teach.",
    "api_token_missing": "the requested API never appeared in the text.",
    "copied_exemplar":
        "it reused the subject of one of the examples. The examples show FORM "
        "only — write about the topic you were given.",
    "harness_marker": "it mentioned benchmark harness markers.",
}

_PLAN_PROMPT = """\
You are planning short knowledge-base documents about the Python library $lib.
They are read by a developer who is completing a partially written snippet: the
imports and the data already exist above the gap, and the developer needs the
missing lines.

A document is worth writing when an EVERYDAY operation has a trap inside it —
two neighbouring calls that look interchangeable and are not, an argument whose
default silently changes the result, or a spelling that was renamed. Write about
operations people use constantly; the trap is what makes the document useful,
not the obscurity of the function.

Evidence collected from this library's own documentation and example code:
$evidence

Already planned — do not propose these again, or the same thing in other words:
$avoid

Propose up to $n NEW documents. For each one:
- pick the operation from the evidence above;
- name the exact confusion in one line;
- name the API token the document will be about;
- name the variables the reader is assumed to already have.

Answer with ONE JSON object per line and nothing else:
{"lib": "$lib", "topic": "<the question the user would type>", "trap": "<what
exactly gets confused, one line>", "api": "<the API token>", "vars": "<variables
the reader already has, comma separated>"}
"""

_RECIPE_PROMPT = """\
You are writing one document for a knowledge base that is read by a developer
who is completing a partially written $lib snippet. The imports and the data are
already in place above the gap; the developer needs the missing lines and
nothing else.

Purpose of this document: settle ONE confusion so the reader picks the right
call the first time. It is a directive note, not a tutorial: if the task wants
THIS, use THAT.

The question it answers: $topic
The confusion it must settle: $trap
The API it is about: $api
Variables the reader already has: $vars

Form:
$rules- under 120 words in total.

$examples
$selfcheck
Answer STRICTLY as:
TITLE: <the question, containing $api>
<the document>
"""
