"""ApiGuideGenerator — API-anchored instruction generation, repaired for exp17.

The exp15 version scored 0.430 and its post-mortem found two independent
defects, one mechanical and one conceptual.

**Mechanical.** The bare-method regex also matched the tail of a qualified call,
so ``np.arange(`` was counted both as ``np.arange`` and as a method ``.arange``
belonging to whatever library the surrounding DOCUMENT was about. 41 326 of the
87 341 bare-method matches (47%) were such tails, and 39 of the 149 selected
APIs ended up under the wrong library: ``(scipy, .plot)`` with 1065 hits,
``(matplotlib, .arange)`` with 384, ``(pandas, .randn)`` with 233. The corpus
duly produced a document titled "How do I use the numpy method .plot()", which
then landed in 89 task contexts.

**Conceptual.** Selecting the top-150 APIs by corpus frequency selects exactly
what the model already knows: the median corpus frequency of an API mentioned by
the exp15 generator was 398, against 31 for the manual recipes that actually
moved the metric.

The repair:

* the census skips matches falling inside a qualified call, resolves a bare
  method to the library that owns that name globally (rather than to the library
  of the document it was found in), and merges a bare method into its qualified
  spelling when one exists — cross-library, not per-library as before;
* selection targets the specificity band the manual corpus occupies and ranks by
  confusability (a name another library also owns, or a look-alike sibling)
  instead of by raw frequency;
* documents follow the shared contract in ``src.generation.doc_contract`` — the
  same one ``PlanGuideGenerator`` uses, including the identical protocol pack, so
  the exp17 comparison between the two isolates topic selection.

The census functions are module-level because the planning generator consumes
them too.

NB: no ``from __future__ import annotations`` — the registry introspects real
annotation objects.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from src.agent_constructor.core import Document
from src.agent_constructor.generator import Generator
from src.agent_constructor.prompts import PromptDiscoverable
from src.generation.doc_contract import (DOC_CLASS_RECIPE, GenJob,
                                         check_contract, contract_rules,
                                         protocol_jobs, title_of)
from src.generation.rag_guides import _FENCE_RE, _TITLE_RE, _code_ok, _slug
from src.utils import DOCUMENT_SRC_DOCUMENTS
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)

# qualified calls: pd.concat(, np.linalg.norm(, plt.plot(, torch.stack( ...
_QUAL_RE = re.compile(
    r"\b((?:pd|np|plt|sns|tf|pandas|numpy|scipy|sklearn|torch|tensorflow|"
    r"matplotlib)(?:\.[A-Za-z_]\w*)+)\s*\(")
# bare method calls: .groupby(, .reshape(, .fit_transform( ...
_METH_RE = re.compile(r"\.([a-z_]\w{2,})\s*\(")

_PREFIX_LIB = {
    "pd": "pandas", "pandas": "pandas",
    "np": "numpy", "numpy": "numpy",
    "plt": "matplotlib", "matplotlib": "matplotlib", "sns": "matplotlib",
    "tf": "tensorflow", "tensorflow": "tensorflow",
    "torch": "torch", "scipy": "scipy", "sklearn": "sklearn",
}

# generic python noise we never want an instruction about
_METH_STOP = {
    "print", "len", "range", "type", "int", "str", "float", "list", "dict",
    "set", "tuple", "abs", "open", "input", "enumerate", "zip", "map",
    "filter", "sorted", "isinstance", "getattr", "setattr", "super", "format",
    "items", "keys", "values", "get", "add", "update", "copy", "pop",
    "startswith", "endswith", "strip", "lower", "upper", "replace",
    "remove", "insert", "index", "count", "extend", "clear",
}


# ---------------------------------------------------------------------- census
def api_census(documents: List[Document], path_to_db: str = "",
               stats: Optional[Counter] = None) -> Counter:
    """Count (library, api_token) usages across corpus documents and examples.

    Attribution rules, both of which the exp15 version got wrong:

    * a bare method that is the tail of a qualified call in the same text is not
      counted separately — it is the same API written differently;
    * a bare method is attributed to the library that owns the name globally
      (the library whose qualified calls end with it, or failing that the
      library where the method is most used), never to the library of the
      document it happened to appear in.
    """
    qualified: Counter = Counter()
    bare: Dict[str, Counter] = defaultdict(Counter)   # method -> lib -> count
    skipped_tails = 0

    def scan(text: str, lib: Optional[str]) -> None:
        nonlocal skipped_tails
        spans = []
        for m in _QUAL_RE.finditer(text):
            token = m.group(1)
            root = token.split(".")[0]
            qualified[(_PREFIX_LIB.get(root, root), token)] += 1
            spans.append((m.start(), m.end()))
        if not lib:
            return
        for m in _METH_RE.finditer(text):
            method = m.group(1)
            if method in _METH_STOP:
                continue
            if any(s <= m.start() < e for s, e in spans):
                skipped_tails += 1        # ".arange" inside "np.arange("
                continue
            bare[method][lib] += 1

    for doc in documents:
        scan(doc.text or "", (doc.metadata or {}).get("library"))

    if path_to_db:
        try:
            con = sqlite3.connect(path_to_db)
            rows = con.execute(
                "SELECT e.content, l.name FROM examples e "
                "JOIN documents d ON e.doc_id = d.id "
                "JOIN sections s ON d.section_id = s.id "
                "JOIN libraries l ON s.library_id = l.id").fetchall()
            con.close()
            for content, lib in rows:
                scan(content or "", lib)
            if stats is not None:
                stats["census_examples"] = len(rows)
        except Exception as exc:
            logger.warning("examples census failed (%s): %s", path_to_db, exc)

    # Who owns each name: the library whose qualified calls end with it.
    owner: Dict[str, Tuple[str, int]] = {}
    for (lib, token), count in qualified.items():
        tail = token.rsplit(".", 1)[-1]
        if tail not in owner or count > owner[tail][1]:
            owner[tail] = (lib, count)

    counts = Counter(qualified)
    merged = 0
    for method, per_lib in bare.items():
        total = sum(per_lib.values())
        if method in owner:
            # Same API as the qualified spelling — merge, do not invent a
            # second entry under another library.
            owner_lib = owner[method][0]
            qual_token = next(
                (t for (l, t) in qualified if l == owner_lib
                 and t.rsplit(".", 1)[-1] == method), None)
            if qual_token is not None:
                counts[(owner_lib, qual_token)] += total
                merged += 1
                continue
        # An object method with no qualified spelling (.groupby, .reshape):
        # attribute it to the library where it is used most, corpus-wide.
        best_lib = max(sorted(per_lib), key=lambda l: per_lib[l])
        counts[(best_lib, "." + method)] += total

    if stats is not None:
        stats["census_skipped_tails"] = skipped_tails
        stats["census_merged_methods"] = merged
        stats["census_apis"] = len(counts)
    logger.info("api census: %d distinct APIs (%d bare-method tails skipped, "
                "%d methods merged into their qualified spelling)",
                len(counts), skipped_tails, merged)
    return counts


def rare_api_candidates(census: Counter, min_count: int = 5,
                        max_count: int = 100) -> List[Tuple[str, str, int]]:
    """APIs in the specificity band the manual recipes occupy (median 31)."""
    out = [(lib, api, count) for (lib, api), count in census.items()
           if min_count <= count <= max_count and len(api.lstrip(".")) > 2]
    out.sort(key=lambda x: (-x[2], x[0], x[1]))
    return out


def _confusability(census: Counter) -> Dict[Tuple[str, str], int]:
    """Score how easy an API is to mix up with a neighbour.

    Two corpus-derived signals: the same short name is owned by more than one
    library, and a same-library sibling shares a stem (argsort/sort,
    trapz/trapezoid). Both are exactly what the manual recipes disambiguate.
    """
    by_name: Dict[str, set] = defaultdict(set)
    by_lib: Dict[str, List[str]] = defaultdict(list)
    for lib, api in census:
        name = api.rsplit(".", 1)[-1].lstrip(".")
        by_name[name].add(lib)
        by_lib[lib].append(name)
    score: Dict[Tuple[str, str], int] = {}
    for lib, api in census:
        name = api.rsplit(".", 1)[-1].lstrip(".")
        points = 1 if len(by_name[name]) > 1 else 0
        siblings = by_lib[lib]
        if any(other != name and len(other) > 4 and len(name) > 4
               and (other.startswith(name) or name.startswith(other))
               for other in siblings):
            points += 1
        score[(lib, api)] = points
    return score


def select_apis(census: Counter, n_docs: int, min_count: int = 5,
                max_count: int = 100) -> List[Tuple[str, str, int]]:
    """Pick the APIs to write about: specific first, confusable first."""
    candidates = rare_api_candidates(census, min_count, max_count)
    if not candidates:
        return []
    conf = _confusability(census)
    per_lib: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    for lib, api, count in candidates:
        per_lib[lib].append((lib, api, count))
    for lib in per_lib:
        per_lib[lib].sort(key=lambda x: (-conf.get((x[0], x[1]), 0), -x[2], x[1]))
    libs = sorted(per_lib)
    quota = max(1, n_docs // max(1, len(libs)))
    picked: List[Tuple[str, str, int]] = []
    for lib in libs:
        picked.extend(per_lib[lib][:quota])
    # Fill any remainder round-robin so a small library does not waste budget.
    if len(picked) < n_docs:
        chosen = set(picked)
        rest = [c for lib in libs for c in per_lib[lib][quota:]
                if c not in chosen]
        rest.sort(key=lambda x: (-conf.get((x[0], x[1]), 0), -x[2], x[1]))
        picked.extend(rest[: n_docs - len(picked)])
    picked.sort(key=lambda x: (x[0], -x[2], x[1]))
    return picked[:n_docs]


# ------------------------------------------------------------------- generator
class ApiGuideGenerator(PromptDiscoverable, Generator):
    """One instruction document per selected corpus API.

    Args:
        url / model_name: the writing model's endpoint and id.
        path_to_db: sqlite path — its ``examples`` table joins the census.
        n_api_docs: total document budget (protocol pack included).
        n_protocol_docs: how much of the budget the protocol class takes.
        min_count / max_count: corpus-frequency window an API must fall into.
        num_workers / temperature / seed / max_retries / limit / max_doc_chars:
            as in ``PlanGuideGenerator``.
        dump_path: audit jsonl of everything that survived.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        path_to_db: str = "",
        n_api_docs: int = 150,
        n_protocol_docs: int = 24,
        min_count: int = 5,
        max_count: int = 100,
        num_workers: int = 8,
        temperature: float = 0.2,
        seed: int = 0,
        max_retries: int = 2,
        limit: int = 0,
        max_doc_chars: int = 1100,
        dump_path: str = "",
        name: str = "api_guide_generator",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.endpoint_url = url or ""
        self.model_name = model_name
        self.path_to_db = path_to_db
        self.n_api_docs = int(n_api_docs)
        self.n_protocol_docs = int(n_protocol_docs)
        self.min_count = max(1, int(min_count))
        self.max_count = int(max_count)
        self.num_workers = max(1, int(num_workers))
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.max_retries = max(0, int(max_retries))
        self.limit = int(limit)
        self.max_doc_chars = int(max_doc_chars)
        self.dump_path = dump_path

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
    def _chat(self, user: str, seed: Optional[int] = None) -> str:
        messages = [
            {"role": "system",
             "content": self.prompt("chat_system", (
                 "You are a senior Python engineer writing short, precise "
                 "knowledge-base documents for developers who complete "
                 "partially written code snippets."
             ))},
            {"role": "user", "content": user},
        ]
        return self._complete(messages, self.temperature,
                              self.seed if seed is None else seed)

    # ------------------------------------------------------------------- jobs
    def _api_jobs(self, apis: List[Tuple[str, str, int]]) -> List[GenJob]:
        jobs = []
        for lib, api, count in apis:
            shown = f"the {lib} method `{api}()`" if api.startswith(".") else f"`{api}`"
            prompt = self.render_prompt(
                "api_recipe", _API_PROMPT,
                lib=lib, api=api, shown=shown, count=count,
                rules=contract_rules(DOC_CLASS_RECIPE, api))
            jobs.append(GenJob(
                library=lib, doc_class=DOC_CLASS_RECIPE, prompt=prompt,
                api_token=api, plan_id=f"api:{lib}:{api}", topic=api))
        return jobs

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        self.stats = Counter()
        self.audit_docs = {}
        self.audit_plans = []

        libs = sorted({(d.metadata or {}).get("library") for d in documents}
                      - {None, ""})
        if not libs:
            logger.warning("%s: corpus has no library metadata", self.name)
            return []

        census = api_census(documents, self.path_to_db, self.stats)
        proto = protocol_jobs(libs, self.n_protocol_docs)
        api_budget = max(0, self.n_api_docs - len(proto))
        apis = select_apis(census, api_budget, self.min_count, self.max_count)
        if not apis:
            logger.warning("%s: empty API selection", self.name)
        self.stats["apis_selected"] = len(apis)
        logger.info("%s: %d APIs selected in the %d..%d frequency window "
                    "(top: %s)", self.name, len(apis), self.min_count,
                    self.max_count, [f"{l}:{a}({c})" for l, a, c in apis[:8]])

        jobs = proto + self._api_jobs(apis)
        if self.limit:
            jobs = jobs[: self.limit]
        self.stats["jobs"] = len(jobs)

        produced = self._run_jobs(jobs)
        return self._package(produced)

    # ------------------------------------------------------------ production
    def _write_doc(self, job: GenJob, index: int) -> Optional[Tuple[GenJob, str, int]]:
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
            note = ("\nYour previous answer was rejected (" + violation +
                    "). Rewrite it following the requirements exactly.")
        self.stats["retry_exhausted"] += 1
        self.audit_plans.append({"library": job.library, "status": "retry_exhausted",
                                 "plan_id": job.plan_id, "topic": job.topic, "raw": ""})
        return None

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
            lambda mm: "\n".join("    " + l for l in
                                 mm.group(1).rstrip().split("\n")) + "\n",
            content).strip() + "\n"
        if not (120 <= len(content) <= self.max_doc_chars):
            self.stats["drop_length"] += 1
            return None
        return content

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
                except Exception as exc:
                    self.stats["drop_job_error"] += 1
                    logger.warning("job %d failed: %s", i, exc)
        return [r for r in results if r is not None]

    def _package(self, produced: List[Tuple[GenJob, str, int]]) -> List[Document]:
        seen = set()
        synth: List[Document] = []
        dump_f = None
        if self.dump_path:
            os.makedirs(os.path.dirname(self.dump_path) or ".", exist_ok=True)
            dump_f = open(self.dump_path, "w", encoding="utf-8")
        try:
            for job, content, attempts in produced:
                doc_name = _slug(title_of(content), job.prefix)
                if doc_name in seen:
                    self.stats["drop_dup_output"] += 1
                    self.audit_plans.append(
                        {"library": job.library, "status": "duplicate:output",
                         "plan_id": job.plan_id, "topic": job.topic, "raw": ""})
                    continue
                seen.add(doc_name)
                self.audit_docs[doc_name] = {
                    "doc_class": job.doc_class, "plan_id": job.plan_id,
                    "topic": job.topic, "api_token": job.api_token,
                    "prompt_sha1": hashlib.sha1(
                        job.prompt.encode("utf-8")).hexdigest()[:12],
                    "attempts": attempts}
                self.audit_plans.append(
                    {"library": job.library, "status": "generated",
                     "plan_id": job.plan_id, "topic": job.topic, "raw": ""})
                synth.append(Document(
                    id=f"api_gen_{len(synth)}",
                    text=content,
                    source=DOCUMENT_SRC_DOCUMENTS,
                    metadata={"library": job.library,
                              "section": f"generated_{job.doc_class}",
                              "doc_name": doc_name}))
                if dump_f:
                    dump_f.write(json.dumps(
                        {"library": job.library, "doc_name": doc_name,
                         "stage": job.doc_class, "api": job.api_token,
                         "attempts": attempts, "content": content},
                        ensure_ascii=False) + "\n")
        finally:
            if dump_f:
                dump_f.close()

        statuses = Counter(p["status"] for p in self.audit_plans)
        logger.info(
            "%s: %d documents into the corpus. COVERAGE: jobs %d, generated %d, "
            "lost %d. Statuses: %s. Yield stats: %s",
            self.name, len(synth), self.stats.get("jobs", 0), len(synth),
            max(0, self.stats.get("jobs", 0) - len(synth)), dict(statuses),
            dict(self.stats))
        return synth


_API_PROMPT = """\
Write ONE short knowledge-base document about $shown in modern $lib code, for a \
developer completing a partially written snippet.

The document must resolve a real confusion around $api: a neighbouring API that \
does almost the same thing, an argument whose default silently changes the \
result, or a spelling that was renamed. If $api has no such trap, write about \
the single mistake that actually breaks code using it.

Requirements:
$rules- under 120 words plus the code.

Format STRICTLY as:
TITLE: <a how-do-I question containing $api>
<document text, code in ``` fences>
"""
