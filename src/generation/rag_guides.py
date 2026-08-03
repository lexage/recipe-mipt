"""RagGuideGenerator — deployable corpus-augmentation method (ТЗ 2.1, generation).

Automates the manual DB augmentation that produced ``docs_database_examples_aug*.db``:
from the source corpus alone (no benchmark data of any kind) it generates guide
documents of three kinds ("editorial policies"):

  A. format guides   — how to correctly complete a partially-written snippet
                       (don't repeat setup, don't hardcode example data, assign
                       the requested variable, indented ``def f`` bodies);
  B. migration notes — deprecated/renamed library APIs -> modern replacements.
                       The LLM expands a built-in checklist of deprecation
                       families (universal library knowledge, part of the
                       method), so stale LLM memory does not limit coverage;
  C. how-to recipes  — sampled corpus documents rewritten as retrieval-friendly
                       Q&A recipes (question-style title + short modern-API
                       answer with code).

``policy`` selects the editorial rule-set:
  * ``v1`` — release-notes style migration docs; free-form recipe endings;
             generic format guides.
  * ``v2`` — migration docs as topic Q&A in user-question vocabulary; every
             recipe/code answer must end with an explicit ``result = ...``
             assignment; per-library ``def f`` function-body guides.

Pipeline component for the ``generator`` slot of SIMPLE. Build-time cost is a
few hundred LLM calls, parallelised with ``num_workers``; the thread-local
token tracker is re-armed in workers so usage is counted.

NB: no ``from __future__ import annotations`` — keep annotations concrete for
the registry's dependency introspection.
"""

import ast
import logging
import random
import re
import textwrap
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from src.agent_constructor.core import Document
from src.agent_constructor.generator import Generator
from src.utils import DOCUMENT_SRC_DOCUMENTS
from src.utils.token_tracker import get_active, set_active

logger = logging.getLogger(__name__)

_DOC_SPLIT = re.compile(r"^\s*#{0,3}\s*DOC\s*\d*\s*#*\s*$", re.M | re.I)
_TITLE_RE = re.compile(r"^\s*TITLE\s*:\s*(.+)$", re.M | re.I)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.S)

# ---------------------------------------------------------------------------
# Built-in deprecation checklist (stage B seed). Universal library knowledge —
# the LLM expands each item into a full document; it does not have to recall
# the deprecations itself.
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
    """Generate RAG guide documents from the source corpus.

    Args:
        url: base URL of the OpenAI-compatible LLM endpoint.
        model_name: model id to call.
        policy: editorial rule-set, ``"v1"`` or ``"v2"`` (see module docstring).
        n_format_docs: target number of stage-A format guides.
        n_migration_docs: cap on stage-B migration docs (checklist-driven).
        n_recipe_docs: target number of stage-C how-to recipes.
        num_workers: parallel LLM calls at build time.
        temperature: sampling temperature.
        limit: if > 0, cap LLM calls per stage (smoke runs). 0 = no cap.
        max_doc_chars: drop generated docs longer than this (chunker budget).
        seed: RNG seed for corpus sampling.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        policy: str = "v2",
        n_format_docs: int = 25,
        n_migration_docs: int = 60,
        n_recipe_docs: int = 300,
        num_workers: int = 8,
        temperature: float = 0.6,
        limit: int = 0,
        max_doc_chars: int = 950,
        seed: int = 7,
        name: str = "rag_guide_generator",
    ):
        super().__init__(name)
        if policy not in ("v1", "v2"):
            raise ValueError(f"policy must be 'v1' or 'v2', got {policy!r}")
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.policy = policy
        self.n_format_docs = int(n_format_docs)
        self.n_migration_docs = int(n_migration_docs)
        self.n_recipe_docs = int(n_recipe_docs)
        self.num_workers = max(1, int(num_workers))
        self.temperature = float(temperature)
        self.limit = int(limit)
        self.max_doc_chars = int(max_doc_chars)
        self.seed = int(seed)

    # -------------------------------------------------------------------- LLM
    def _chat(self, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system",
                 "content": "You are a senior Python engineer writing short, "
                            "precise knowledge-base documents."},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    # ---------------------------------------------------------------- parsing
    def _parse_docs(self, text: str, lib: str, prefix: str) -> List[Tuple[str, str, str]]:
        """Split an LLM response into (library, doc_name, content) triples."""
        out = []
        parts = _DOC_SPLIT.split(text)
        candidates = parts[1:] if len(parts) > 1 else parts
        for part in candidates:
            part = part.strip()
            if not part:
                continue
            m = _TITLE_RE.search(part)
            title = m.group(1).strip() if m else part.splitlines()[0][:70]
            body = _TITLE_RE.sub("", part, count=1).strip()
            # keep the title as the first line of the content (embedding anchor)
            content = title + "\n\n" + body
            # replace markdown fences with 4-space indented code
            def _replace(match):
                return textwrap.indent(match.group(1).rstrip(), "    ") + "\n"
            blocks = _FENCE_RE.findall(content)
            if any(not _code_ok(b) for b in blocks):
                continue
            content = _FENCE_RE.sub(_replace, content).strip() + "\n"
            if not (150 <= len(content) <= self.max_doc_chars):
                continue
            out.append((lib, _slug(title, prefix), content))
        return out

    # ------------------------------------------------------------- stage jobs
    def _format_jobs(self, libs: List[str]) -> List[Tuple[str, str, str]]:
        per_lib = max(1, round(self.n_format_docs / max(1, len(libs))))
        common = (
            "Write {k} SHORT knowledge-base documents (each under 120 words plus "
            "at most 8 lines of code) teaching how to correctly complete a "
            "partially-written Python snippet that uses {lib}.\n"
            "Cover across the documents: (1) the setup code shown before the gap "
            "has already been executed — never repeat imports or setup lines and "
            "never call placeholder loaders again; (2) never re-create or "
            "hard-code the example input data — the same code runs on hidden "
            "inputs, derive sizes from the data; (3) assign the final answer to "
            "exactly the variable name the task requests (often `result`); "
            "(4) output raw code only — no tags, no markdown, no prose around "
            "the code. Include one small WRONG-vs-RIGHT example per document "
            "using realistic {lib} objects (invent your own tiny data).\n"
        )
        v2_extra = (
            "One of the documents MUST be dedicated to completing an unfinished "
            "function stub like `def f(data = example_data):` — explain that the "
            "answer is ONLY the indented function body ending with `return`, "
            "with no new `def` header, no imports at column 0 and no call to "
            "the function afterwards; show a tiny correct {lib} body.\n"
        )
        tail = (
            "Format STRICTLY as:\n### DOC\nTITLE: <short how-to title>\n"
            "<document text, code in ``` fences>\n### DOC\n..."
        )
        jobs = []
        for lib in libs:
            prompt = common.format(k=per_lib, lib=lib)
            if self.policy == "v2":
                prompt += v2_extra.format(lib=lib)
            jobs.append((lib, "fmt", prompt + tail))
        return jobs

    def _migration_jobs(self, libs: List[str]) -> List[Tuple[str, str, str]]:
        style_v1 = (
            "For EACH checklist item below write one SHORT release-notes style "
            "migration document: the title states the OLD API name and that it "
            "was removed/renamed; the body names the modern replacement and "
            "shows a 2-5 line code example of the modern call.\n"
        )
        style_v2 = (
            "For EACH checklist item below write one SHORT Q&A document: the "
            "TITLE is a how-do-I question phrased the way a user would describe "
            "the practical TASK (do not mention the old API in the title); the "
            "body gives the modern idiom with a 2-6 line code example that ends "
            "with an assignment like `result = ...` where it makes sense, and "
            "mentions in one sentence that the legacy spelling was removed.\n"
        )
        tail = (
            "Each document under 110 words plus the code. Format STRICTLY as:\n"
            "### DOC\nTITLE: <title>\n<text, code in ``` fences>\n### DOC\n..."
        )
        jobs = []
        budget = self.n_migration_docs
        for lib in libs:
            items = DEPRECATIONS.get(lib, [])
            if not items or budget <= 0:
                continue
            items = items[:budget]
            budget -= len(items)
            for batch_start in range(0, len(items), 4):
                batch = items[batch_start:batch_start + 4]
                checklist = "\n".join(f"- {it}" for it in batch)
                style = style_v1 if self.policy == "v1" else style_v2
                prompt = (f"Library: {lib}.\n" + style +
                          f"Checklist:\n{checklist}\n" + tail)
                jobs.append((lib, "migr", prompt))
        return jobs

    def _recipe_jobs(self, documents: List[Document]) -> List[Tuple[str, str, str]]:
        rng = random.Random(self.seed)
        by_lib = defaultdict(list)
        for d in documents:
            lib = (d.metadata or {}).get("library")
            if lib and len(d.text) > 300:
                by_lib[lib].append(d)
        total = sum(len(v) for v in by_lib.values())
        if not total:
            return []
        docs_per_call = 2
        n_calls = max(1, round(self.n_recipe_docs / docs_per_call))
        jobs = []
        for lib, docs in by_lib.items():
            quota = max(1, round(n_calls * len(docs) / total))
            for d in rng.sample(docs, min(quota, len(docs))):
                fragment = d.text[:1600]
                v2_rule = (
                    " The code MUST end with an explicit assignment of the "
                    "final value to a variable named `result`."
                    if self.policy == "v2" else ""
                )
                prompt = (
                    f"Below is a fragment of {lib} documentation.\n"
                    "-----\n" + fragment + "\n-----\n"
                    f"Rewrite the practical knowledge of this fragment as "
                    f"{docs_per_call} SHORT how-to documents. Each: TITLE is a "
                    "question phrased exactly the way a user would ask it "
                    "(e.g. 'How do I ... ?'), then a 2-4 sentence answer, then "
                    "a minimal code example (2-8 lines) using the modern "
                    f"{lib} API.{v2_rule} Invent your own tiny example data; "
                    "do not copy long text verbatim. Under 120 words each.\n"
                    "Format STRICTLY as:\n### DOC\nTITLE: <question>\n"
                    "<answer with code in ``` fences>\n### DOC\n..."
                )
                jobs.append((lib, "howto", prompt))
        return jobs

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        libs_counter = Counter(
            (d.metadata or {}).get("library") for d in documents)
        libs = sorted(l for l in libs_counter if l)
        if not libs:
            logger.warning("RagGuideGenerator: no library metadata in corpus")
            return []

        jobs = (self._format_jobs(libs)
                + self._migration_jobs(libs)
                + self._recipe_jobs(documents))
        if self.limit:
            by_stage = defaultdict(list)
            for j in jobs:
                by_stage[j[1]].append(j)
            jobs = [j for st in by_stage.values() for j in st[: self.limit]]

        logger.info("RagGuideGenerator(policy=%s): %d LLM jobs (%d libs)",
                    self.policy, len(jobs), len(libs))

        def work(job):
            lib, prefix, prompt = job
            return self._parse_docs(self._chat(prompt), lib, prefix)

        parsed: List[Tuple[str, str, str]] = []
        tracker = get_active()
        with ThreadPoolExecutor(max_workers=self.num_workers,
                                initializer=set_active,
                                initargs=(tracker,)) as ex:
            futures = [ex.submit(work, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{self.name}:{self.policy}"):
                try:
                    parsed.extend(fut.result())
                except Exception as exc:   # one bad call must not kill the run
                    logger.warning("guide generation call failed: %s", exc)

        seen = set()
        ids_offset = len(documents) + 1
        synth_docs: List[Document] = []
        for lib, doc_name, content in parsed:
            if doc_name in seen:
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
        by_prefix = Counter(n.split("_")[0] for _l, n, _c in
                            [(l, n, c) for l, n, c in parsed if n in seen])
        logger.info("RagGuideGenerator: %d docs generated (fmt/migr/howto "
                    "mix: %s), policy=%s", len(synth_docs),
                    dict(by_prefix), self.policy)
        return synth_docs
