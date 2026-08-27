"""ApiGuideGenerator — API-anchored instruction generation, repaired for exp18.

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
from src.generation.doc_contract import (DOC_CLASS_PROTOCOL, DOC_CLASS_RECIPE,
                                         SELF_CHECK, GenJob, check_contract,
                                         contract_rules, exemplar_topics,
                                         exemplars_block, keep_first_code_block,
                                         protocol_jobs, strip_imports, title_of)
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


def is_public(api: str) -> bool:
    """Dunders and private helpers are never worth a document."""
    name = api.lstrip(".")
    return bool(name) and not name.startswith("_") and "._" not in api


def head_api_candidates(census: Counter, min_count: int = 100
                        ) -> List[Tuple[str, str, int]]:
    """APIs from the HEAD of the frequency distribution.

    exp17 selected the rare tail (5..100 uses) because the manual recipes have a
    median API frequency of 31. That reading was wrong: the manual median is a
    median over a MIXTURE — its 75th percentile is 104 and its maximum 2770. The
    manual documents anchor on an everyday operation and put the trap inside it;
    the rare token is a detail of the solution, not the subject. Selecting by
    rarity produced tf.distribute.ReduceOp.SUM and scipy.special.ccdf, and only
    21% of the selected APIs occurred anywhere in user task text against 62% for
    this rule (measured before the run).
    """
    out = [(lib, api, count) for (lib, api), count in census.items()
           if count >= min_count and len(api.lstrip(".")) > 2 and is_public(api)]
    out.sort(key=lambda x: (-x[2], x[0], x[1]))
    return out


def lookalike_siblings(census: Counter, min_count: int = 100,
                       sibling_min: int = 20
                       ) -> Dict[Tuple[str, str], List[Tuple[str, int]]]:
    """For each frequent API, the same-library names it is easy to mix up with.

    The ANCHOR has to be frequent — that is the operation the user is actually
    doing. The neighbour does not: np.roll against a NaN-filling slice, or
    sort_values against sort_index, is exactly the manual-recipe pattern where
    one side of the confusion is much rarer than the other.
    """
    by_lib: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (lib, api), count in census.items():
        if count >= sibling_min and is_public(api):
            by_lib[lib].append((api.rsplit(".", 1)[-1].lstrip("."), count))
    out: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
    for (lib, api), count in census.items():
        if count < min_count or not is_public(api):
            continue
        name = api.rsplit(".", 1)[-1].lstrip(".")
        sibs = []
        for other, ocount in by_lib[lib]:
            if other == name or len(other) < 4 or len(name) < 4:
                continue
            if other.startswith(name) or name.startswith(other) or \
                    (other[:4] == name[:4] and abs(len(other) - len(name)) <= 6):
                sibs.append((other, ocount))
        if sibs:
            sibs.sort(key=lambda x: -x[1])
            out[(lib, api)] = sibs[:3]
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


def select_apis(census: Counter, n_docs: int, min_count: int = 100,
                require_confusable: bool = True) -> List[Tuple[str, str, int]]:
    """Pick the APIs to write about: the head of each library, confusable first.

    Two properties the exp17 version lacked:

    * the head is taken PER LIBRARY, not against one absolute threshold. The
      corpus is lopsided (49 numpy APIs above 100 uses against 3 sklearn ones),
      so a global cut silently hands the whole budget to the biggest library —
      exp17 gave tensorflow 24-30 documents for 45 benchmark tasks and sklearn
      2-3 for 115;
    * confusability ranks candidates instead of filtering them out, so a library
      with few confusable APIs still fills its quota rather than going empty.
    """
    conf = _confusability(census)
    per_lib: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    for (lib, api), count in census.items():
        if len(api.lstrip(".")) > 2 and is_public(api):
            per_lib[lib].append((lib, api, count))
    if not per_lib:
        return []

    def rank(item):
        lib, api, count = item
        return (-(conf.get((lib, api), 0) if require_confusable else 0), -count, api)

    libs = sorted(per_lib)
    quota = max(1, n_docs // len(libs))
    picked: List[Tuple[str, str, int]] = []
    leftovers: List[Tuple[str, str, int]] = []
    for lib in libs:
        ranked = sorted(per_lib[lib], key=rank)
        # The head of THIS library: everything above the absolute threshold,
        # and, if that is short of the quota, its next most used APIs — but
        # never below the floor. A library with a thin head gets fewer
        # documents rather than documents about APIs used twice, which is the
        # exp17 failure mode reappearing through the back door.
        head = [c for c in ranked if c[2] >= min_count]
        if len(head) < quota:
            floor = max(2, min_count // 5)
            head += [c for c in ranked if floor <= c[2] < min_count][
                :quota - len(head)]
        picked.extend(head[:quota])
        leftovers.extend(head[quota:])
    if len(picked) < n_docs:
        leftovers.sort(key=rank)
        picked.extend(leftovers[: n_docs - len(picked)])
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
        min_count: an API must be used at least this many times in the corpus
            (the HEAD of the distribution, see head_api_candidates).
        fewshot_db_path / n_fewshot: manual corpus used as FORM exemplars.
        reject_dump_path: where rejected drafts go, so a shrinking yield can be
            audited instead of guessed at.
        num_workers / temperature / seed / max_retries / limit / max_doc_chars:
            as in ``PlanGuideGenerator``.
        dump_path: audit jsonl of everything that survived.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        path_to_db: str = "",
        n_api_docs: int = 220,
        n_protocol_docs: int = 70,
        min_count: int = 100,
        fewshot_db_path: str = "data/docs_database_examples_aug.db",
        n_fewshot: int = 3,
        num_workers: int = 8,
        temperature: float = 0.2,
        seed: int = 0,
        max_retries: int = 2,
        limit: int = 0,
        max_doc_chars: int = 1100,
        dump_path: str = "",
        reject_dump_path: str = "",
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
        self.fewshot_db_path = fewshot_db_path
        self.n_fewshot = int(n_fewshot)
        self.num_workers = max(1, int(num_workers))
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.max_retries = max(0, int(max_retries))
        self.limit = int(limit)
        self.max_doc_chars = int(max_doc_chars)
        self.dump_path = dump_path
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
        for i, (lib, api, count) in enumerate(apis):
            shown = f"the {lib} method `{api}()`" if api.startswith(".") else f"`{api}`"
            prompt = self.render_prompt(
                "api_recipe", _API_PROMPT,
                lib=lib, api=api, shown=shown, count=count,
                rules=contract_rules(DOC_CLASS_RECIPE, api),
                examples=exemplars_block(self.fewshot_db_path, DOC_CLASS_RECIPE,
                                         rotation=i, n=self.n_fewshot),
                selfcheck=SELF_CHECK)
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
        proto = protocol_jobs(libs, self.n_protocol_docs,
                              render=self.render_prompt,
                              fewshot_db_path=self.fewshot_db_path,
                              n_fewshot=self.n_fewshot)
        api_budget = max(0, self.n_api_docs - len(proto))
        apis = select_apis(census, api_budget, self.min_count)
        if not apis:
            logger.warning("%s: empty API selection", self.name)
        self.stats["apis_selected"] = len(apis)
        logger.info("%s: %d APIs selected from the head of the distribution "
                    "(>=%d uses, confusable) (top: %s)", self.name, len(apis),
                    self.min_count, [f"{l}:{a}({c})" for l, a, c in apis[:8]])

        jobs = proto + self._api_jobs(apis)
        if self.limit:
            jobs = jobs[: self.limit]
        self.stats["jobs"] = len(jobs)

        produced = self._run_jobs(jobs)
        return self._package(produced)

    # ------------------------------------------------------------ production
    def _write_doc(self, job: GenJob, index: int) -> Optional[Tuple[GenJob, str, int]]:
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

        if self.reject_dump_path and self.rejects:
            os.makedirs(os.path.dirname(self.reject_dump_path) or ".", exist_ok=True)
            with open(self.reject_dump_path, "w", encoding="utf-8") as f:
                for row in self.rejects:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        statuses = Counter(p["status"] for p in self.audit_plans)
        logger.info(
            "%s: %d documents into the corpus. COVERAGE: jobs %d, generated %d, "
            "lost %d. Statuses: %s. Yield stats: %s",
            self.name, len(synth), self.stats.get("jobs", 0), len(synth),
            max(0, self.stats.get("jobs", 0) - len(synth)), dict(statuses),
            dict(self.stats))
        return synth


_API_PROMPT = """\
You are writing one document for a knowledge base that is read by a developer
who is completing a partially written $lib snippet. The imports and the data are
already in place above the gap; the developer needs the missing lines and
nothing else.

Purpose of this document: stop ONE specific confusion around $shown. It is a
directive note, not a tutorial: if the task wants THIS, use THAT.

The subject: $shown, used $count times across this library's own documentation.
Write about the neighbouring API that is easy to take for it, or about the
argument whose default silently changes the result. If neither is true of $api,
write about the single mistake that actually breaks code using it.

Form:
$rules- under 120 words in total.

$examples
$selfcheck
Answer STRICTLY as:
TITLE: <a how-do-I question containing $api>
<the document>
"""
