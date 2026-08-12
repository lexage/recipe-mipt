"""RagGuideGenerator — deployable corpus-augmentation method (ТЗ 2.1, generation).

Instruction-first redesign after exp13. The exp13 variant spent 88% of its
budget rewriting random corpus fragments into Q&A recipes; those covered
topics unrelated to what users actually ask, gave no lift and mildly displaced
real documentation. What DID move metrics (both manually in exp12 and in the
generated runs) were behavioural INSTRUCTIONS. So the method now generates
instructions and prompts only — it never paraphrases the corpus:

  A. behaviour instructions — how to correctly complete a partially-written
     snippet (don't repeat setup, don't hardcode example data, assign the
     requested variable, indented ``def f`` bodies, no wrappers). Many short
     variants per library: retrieval is a lottery, phrasing variety buys
     coverage.
  T. task-type instructions — the LLM first enumerates the most common
     practical "how do I ...?" questions per library (its own knowledge, no
     benchmark), then writes one instruction per topic: recommended modern
     approach, the common mistake to avoid, small code example. The corpus is
     only used to weight libraries; its text is never rewritten.
  B. migration instructions — deprecated/renamed API -> modern replacement,
     expanded from a built-in checklist of deprecation families.

``policy`` selects the editorial rule-set:
  * ``v1`` — release-notes style migration docs; free-form endings.
  * ``v2`` — migration docs as topic Q&A in user-question vocabulary; code
             examples must end with an explicit ``result = ...`` assignment;
             extra per-library ``def f`` function-body instruction angles.

Audit: every surviving document is appended to ``dump_path`` (jsonl) when the
parameter is set, and per-stage yield/drop counters are logged.

Pipeline component for the ``generator`` slot of SIMPLE. Parallel LLM calls
(``num_workers``) with the thread-local token tracker re-armed in workers.

NB: no ``from __future__ import annotations`` — keep annotations concrete for
the registry's dependency introspection.
"""

import ast
import json
import logging
import os
import re
import textwrap
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from src.agent_constructor.core import Document
from src.agent_constructor.generator import Generator
from src.agent_constructor.prompts import PromptDiscoverable
from src.utils import DOCUMENT_SRC_DOCUMENTS
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)

_TITLE_RE = re.compile(r"^\s*TITLE\s*:\s*(.+)$", re.M | re.I)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.S)
_TOPIC_RE = re.compile(r"^\s*[-*\d.)\s]*\s*(.+?)\s*$")

# ---------------------------------------------------------------------------
# Stage-A instruction angles. Each angle -> many phrasing variants (the LLM is
# asked to write ONE short document per call, from one angle, for one library).
# ---------------------------------------------------------------------------
ANGLES_V1 = [
    ("setup", "the setup code shown before the gap has already been executed: "
              "never repeat imports or data definitions, never call "
              "placeholder loader functions again — just continue the code"),
    ("hardcode", "never re-create or hard-code the example input data: the "
                 "same completion runs on hidden inputs, so reuse the given "
                 "variables and derive sizes from the data itself"),
    ("assign", "the final answer must be assigned to exactly the variable "
               "name the task requests (often `result`); a bare expression, "
               "a print() or a differently named variable scores nothing"),
    ("rawcode", "output raw executable code only: no XML/markdown wrappers, "
                "no prose around the code, no explanations in other languages"),
]
ANGLES_V2 = ANGLES_V1 + [
    ("defbody", "when the task shows an unfinished function stub like "
                "`def f(data = example):`, the answer is ONLY the indented "
                "function body ending with `return`; no new def header, no "
                "column-0 imports, no call to the function afterwards"),
    ("onlyasked", "do exactly what is asked and nothing extra: no unrequested "
                  "reset_index/astype/sorting/printing; return the object of "
                  "exactly the requested type, name and shape"),
]

# ---------------------------------------------------------------------------
# Stage-B deprecation checklist (universal library knowledge, part of the
# method — the LLM expands items, it does not have to recall them).
# ---------------------------------------------------------------------------
DEPRECATIONS = {
    "numpy": [
        "np.NAN and np.NaN removed -> np.nan (numpy 2.0); np.Inf/np.infty -> np.inf",
        "np.in1d removed -> np.isin",
        "np.math removed -> import the standard math module",
        "np.float/np.int/np.bool/np.object/np.str aliases removed -> builtin types or explicit dtypes",
        "np.alltrue/np.sometrue removed -> np.all/np.any",
        "np.product/np.cumproduct/np.round_ removed -> np.prod/np.cumprod/np.round",
        "np.trapz renamed -> np.trapezoid",
        "np.row_stack/np.msort removed -> np.vstack/np.sort(axis=0)",
        "np.mat removed; np.matrix discouraged -> 2-D ndarray with @ operator",
    ],
    "pandas": [
        "DataFrame.append/Series.append removed -> pd.concat([...])",
        "DataFrame.applymap removed -> DataFrame.map",
        "Series.iteritems/DataFrame.iteritems removed -> .items()",
        "read_csv(delim_whitespace=True) removed -> sep=r'\\s+'",
        "fillna(method='ffill'/'bfill') removed -> .ffill()/.bfill()",
        "replace(..., method='ffill') removed -> replace to NaN then .ffill()",
        "set_axis/rename inplace= keyword removed -> assign the result back",
        "DataFrame.lookup removed -> index.get_indexer + numpy fancy indexing",
        "top-level pd.value_counts removed -> Series.value_counts method",
        "Series.dt.week/weekofyear removed -> dt.isocalendar().week",
        "frequency aliases 'T'->'min', 'S'->'s', 'H'->'h', 'M'->'ME', 'A'->'YE' (resample/date_range/rolling)",
        "groupby.agg([min, max]) builtin names -> pass strings ['min','max'] for stable column names",
        "str.split/str.rsplit max-splits argument is keyword-only -> n=1",
        "DataFrame.drop positional axis removed -> drop(columns=...) or axis= keyword",
        "strict string dtype: cannot assign ints/floats into a 'str' column -> replace whole column or cast astype(object) first",
        "reductions on object/string columns raise -> numeric_only=True or select_dtypes/convert before mean/sum",
        "rounding columns containing pd.NA raises in float() -> apply round only to notna cells or astype('Float64').round",
        "pd.to_datetime matches format exactly -> fix the format string or use format='mixed'",
    ],
    "scipy": [
        "scipy.integrate.simps/trapz/cumtrapz removed -> simpson/trapezoid/cumulative_trapezoid",
        "scipy.interpolate.interp2d removed -> RegularGridInterpolator or RectBivariateSpline",
        "scipy.integrate.quadrature/romberg removed -> quad or fixed_quad",
        "sparse matrix .A attribute removed -> .toarray()",
        "scipy.misc.face/ascent/imread removed -> scipy.datasets / imageio",
        "find_simplex belongs to spatial.Delaunay, not Voronoi -> Delaunay(points).find_simplex",
    ],
    "sklearn": [
        "AgglomerativeClustering affinity= renamed -> metric= (precomputed distances: metric='precomputed', linkage != 'ward')",
        "OneHotEncoder sparse= renamed -> sparse_output=",
        "get_feature_names removed -> get_feature_names_out",
        "estimators require 2-D X -> reshape(-1, 1) single features",
        "mixed-type column names rejected -> df.columns.astype(str) or fit on .to_numpy()",
    ],
    "matplotlib": [
        "seaborn catplot kind='scatter' invalid -> kind='strip' (categorical) or relplot for numeric scatter",
        "seaborn distplot removed -> histplot / displot / kdeplot",
    ],
    "torch": [
        "ByteTensor masks deprecated -> index with .bool() masks; index tensors must be .long()",
    ],
    "tensorflow": [
        "tf.Session/TF1 graph style removed -> eager TF2 ops, @tf.function for graphs",
    ],
}


def _slug(title: str, prefix: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60]
    return f"{prefix}_{s or 'doc'}"


def _code_ok(code: str) -> bool:
    """AST-validate a code block; tolerate indented function-body snippets."""
    code = textwrap.dedent(code).strip()
    if not code:
        return True
    for candidate in (code, "def _wrap_():\n" + textwrap.indent(code, "    ")):
        try:
            ast.parse(candidate)
            return True
        except SyntaxError:
            continue
    return False


class RagGuideGenerator(Generator):
    """Generate instruction documents for RAG from the source corpus + LLM.

    Args:
        url: base URL of the OpenAI-compatible LLM endpoint.
        model_name: model id to call.
        policy: editorial rule-set, ``"v1"`` or ``"v2"`` (see module docstring).
        n_format_docs: target number of stage-A behaviour instructions.
        n_tasktype_docs: target number of stage-T task-type instructions.
        n_migration_docs: cap on stage-B migration instructions.
        num_workers: parallel LLM calls at build time.
        temperature: sampling temperature.
        limit: if > 0, cap LLM calls per stage (smoke runs). 0 = no cap.
        max_doc_chars: drop generated docs longer than this (chunker budget).
        dump_path: if set, append surviving docs to this jsonl file (audit).
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        policy: str = "v2",
        n_format_docs: int = 90,
        n_tasktype_docs: int = 120,
        n_migration_docs: int = 45,
        num_workers: int = 8,
        temperature: float = 0.7,
        limit: int = 0,
        max_doc_chars: int = 1100,
        dump_path: str = "",
        name: str = "rag_guide_generator",
    ):
        super().__init__(name)
        if policy not in ("v1", "v2"):
            raise ValueError(f"policy must be 'v1' or 'v2', got {policy!r}")
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.policy = policy
        self.n_format_docs = int(n_format_docs)
        self.n_tasktype_docs = int(n_tasktype_docs)
        self.n_migration_docs = int(n_migration_docs)
        self.num_workers = max(1, int(num_workers))
        self.temperature = float(temperature)
        self.limit = int(limit)
        self.max_doc_chars = int(max_doc_chars)
        self.dump_path = dump_path
        self.stats = Counter()

    # -------------------------------------------------------------------- LLM
    def _chat(self, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system",
                 "content": self.prompt("chat_system", (
                     "You are a senior Python engineer writing short, "
                     "precise knowledge-base documents."
                 ))},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    # ---------------------------------------------------------------- parsing
    def _parse_doc(self, text: str, lib: str, prefix: str) -> Optional[Tuple[str, str, str]]:
        """Parse ONE document from an LLM response; None if invalid."""
        text = text.strip()
        if not text:
            self.stats["drop_empty"] += 1
            return None
        m = _TITLE_RE.search(text)
        title = m.group(1).strip() if m else text.splitlines()[0][:70]
        body = _TITLE_RE.sub("", text, count=1).strip()
        content = title + "\n\n" + body

        blocks = _FENCE_RE.findall(content)
        if any(not _code_ok(b) for b in blocks):
            self.stats["drop_bad_code"] += 1
            return None

        def _replace(match):
            return textwrap.indent(match.group(1).rstrip(), "    ") + "\n"
        content = _FENCE_RE.sub(_replace, content).strip() + "\n"
        if len(content) < 120:
            self.stats["drop_too_short"] += 1
            return None
        if len(content) > self.max_doc_chars:
            self.stats["drop_too_long"] += 1
            return None
        self.stats["parsed_ok"] += 1
        return (lib, _slug(title, prefix), content)

    # ------------------------------------------------------------- stage jobs
    def _format_jobs(self, libs: List[str]) -> List[Tuple[str, str, str]]:
        angles = ANGLES_V1 if self.policy == "v1" else ANGLES_V2
        per_lib = max(1, round(self.n_format_docs / max(1, len(libs))))
        result_rule = (
            " If the document contains example code producing a value, the "
            "code must end with an assignment like `result = ...`."
            if self.policy == "v2" else ""
        )
        jobs = []
        for lib in libs:
            for i in range(per_lib):
                key, angle = angles[i % len(angles)]
                variant = i // len(angles) + 1
                prompt = (
                    f"Write ONE short knowledge-base document (under 130 words "
                    f"plus at most 8 lines of code) for developers who complete "
                    f"partially-written Python snippets that use {lib}.\n"
                    f"The single rule this document must teach: {angle}.\n"
                    f"Use realistic {lib} objects in a tiny WRONG-vs-RIGHT "
                    f"example (invent your own data). Phrase the title and text "
                    f"in your own words, variation #{variant} — do not use a "
                    f"generic title like 'Instructions'.{result_rule}\n"
                    "Format STRICTLY as:\nTITLE: <short specific title>\n"
                    "<document text, code in ``` fences>"
                )
                jobs.append((lib, f"fmt_{key}", prompt))
        return jobs

    def _migration_jobs(self, libs: List[str]) -> List[Tuple[str, str, str]]:
        style_v1 = (
            "write one SHORT release-notes style migration document: the title "
            "states the OLD API name and that it was removed/renamed; the body "
            "names the modern replacement and shows a 2-5 line example of the "
            "modern call.\n"
        )
        style_v2 = (
            "write one SHORT Q&A document: the TITLE is a how-do-I question "
            "phrased the way a user would describe the practical TASK (do not "
            "mention the old API in the title); the body gives the modern "
            "idiom with a 2-6 line code example that ends with an assignment "
            "like `result = ...` where it makes sense, and mentions in one "
            "sentence that the legacy spelling was removed.\n"
        )
        jobs = []
        budget = self.n_migration_docs
        for lib in libs:
            items = DEPRECATIONS.get(lib, [])
            if not items or budget <= 0:
                continue
            items = items[:budget]
            budget -= len(items)
            style = style_v1 if self.policy == "v1" else style_v2
            for item in items:
                prompt = (f"Library: {lib}. For the deprecation fact below, "
                          + style +
                          f"Fact: {item}\n"
                          "Under 110 words plus the code. Format STRICTLY "
                          "as:\nTITLE: <title>\n<text, code in ``` fences>")
                jobs.append((lib, "migr", prompt))
        return jobs

    def _topic_lists(self, lib_quota: List[Tuple[str, int]]) -> List[Tuple[str, str]]:
        """Phase 1 of stage T: enumerate common practical questions per library."""
        pairs = []
        for lib, quota in lib_quota:
            prompt = (
                f"List the {quota} most common practical questions that "
                f"working developers ask about the Python library {lib} — "
                "everyday operations (creating/transforming/selecting/"
                "combining data, common pitfalls), not exotic features.\n"
                "One question per line, phrased as the user would ask it "
                "('How do I ...?'). No numbering comments, no explanations."
            )
            try:
                text = self._chat(prompt)
            except Exception as exc:
                logger.warning("topic enumeration failed for %s: %s", lib, exc)
                continue
            topics = []
            for line in text.splitlines():
                mt = _TOPIC_RE.match(line)
                t = (mt.group(1) if mt else line).strip()
                if len(t) > 15 and "?" in t:
                    topics.append(t)
            self.stats[f"topics_{lib}"] = len(topics[:quota])
            pairs.extend((lib, t) for t in topics[:quota])
        return pairs

    def _tasktype_jobs(self, topic_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        result_rule = (
            " The code MUST end with an explicit assignment of the final "
            "value to a variable named `result`."
            if self.policy == "v2" else ""
        )
        jobs = []
        for lib, topic in topic_pairs:
            prompt = (
                f"Write ONE short instruction document for Python tasks about: "
                f"\"{topic}\" ({lib}).\n"
                "Structure: TITLE repeats the question in the user's words; "
                "then 2-4 sentences: the recommended modern approach and the "
                "common mistake to avoid; then a minimal code example "
                f"(2-8 lines) with the modern {lib} API using your own tiny "
                f"invented data.{result_rule} Under 120 words plus the code.\n"
                "Format STRICTLY as:\nTITLE: <the question>\n"
                "<text, code in ``` fences>"
            )
            jobs.append((lib, "task", prompt))
        return jobs

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        self.stats = Counter()
        libs_counter = Counter(
            (d.metadata or {}).get("library") for d in documents)
        libs = sorted(l for l in libs_counter if l)
        if not libs:
            logger.warning("RagGuideGenerator: no library metadata in corpus")
            return []

        # stage-T quotas: proportional to corpus size per library
        total = sum(libs_counter[l] for l in libs)
        lib_quota = [(l, max(3, round(self.n_tasktype_docs * libs_counter[l] / total)))
                     for l in libs]
        if self.limit:
            lib_quota = [(l, min(q, self.limit)) for l, q in lib_quota]

        logger.info("RagGuideGenerator(policy=%s): enumerating task topics "
                    "for %d libraries", self.policy, len(libs))
        topic_pairs = self._topic_lists(lib_quota)

        jobs = (self._format_jobs(libs)
                + self._migration_jobs(libs)
                + self._tasktype_jobs(topic_pairs))
        if self.limit:
            by_stage = {}
            for j in jobs:
                by_stage.setdefault(j[1].split("_")[0], []).append(j)
            jobs = [j for st in by_stage.values() for j in st[: self.limit]]

        logger.info("RagGuideGenerator: %d doc jobs (fmt/migr/task)", len(jobs))
        parsed = self._run_jobs(jobs)
        return self._package(parsed, documents)

    # --------------------------------------------------------------- helpers
    def _run_jobs(self, jobs: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Execute (lib, prefix, prompt) jobs in parallel; return parsed docs."""
        def work(job):
            lib, prefix, prompt = job
            return self._parse_doc(self._chat(prompt), lib, prefix)

        parsed: List[Tuple[str, str, str]] = []
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as ex:
            futures = [ex.submit(work, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{self.name}:{self.policy}"):
                try:
                    doc = fut.result()
                    if doc is not None:
                        parsed.append(doc)
                except Exception as exc:   # one bad call must not kill the run
                    self.stats["drop_call_error"] += 1
                    logger.warning("guide generation call failed: %s", exc)
        return parsed

    def _package(self, parsed: List[Tuple[str, str, str]],
                 documents: List[Document]) -> List[Document]:
        """Dedup, wrap into Documents, dump to jsonl, log yield stats."""
        seen = set()
        ids_offset = len(documents) + 1
        synth_docs: List[Document] = []
        dump_f = None
        if self.dump_path:
            os.makedirs(os.path.dirname(self.dump_path) or ".", exist_ok=True)
            dump_f = open(self.dump_path, "w", encoding="utf-8")
        try:
            for lib, doc_name, content in parsed:
                if doc_name in seen:
                    self.stats["drop_dup_name"] += 1
                    continue
                seen.add(doc_name)
                synth_docs.append(
                    Document(
                        id=str(len(synth_docs) + ids_offset),
                        text=content,
                        source=DOCUMENT_SRC_DOCUMENTS,
                        metadata={
                            "library": lib,
                            "section": "rag_guides_gen",
                            "doc_name": doc_name,
                        },
                    )
                )
                if dump_f:
                    dump_f.write(json.dumps(
                        {"library": lib, "doc_name": doc_name,
                         "stage": doc_name.split("_")[0], "content": content},
                        ensure_ascii=False) + "\n")
        finally:
            if dump_f:
                dump_f.close()

        by_stage = Counter(n.split("_")[0] for _l, n, _c in
                           [(l, n, c) for l, n, c in parsed if n in seen])
        logger.info(
            "RagGuideGenerator: %d docs into index (by stage: %s); "
            "yield stats: %s; policy=%s%s",
            len(synth_docs), dict(by_stage), dict(self.stats), self.policy,
            f"; dump={self.dump_path}" if self.dump_path else "")
        return synth_docs
