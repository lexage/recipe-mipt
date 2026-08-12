"""ApiGuideGenerator — API-anchored instruction generation (exp15).

Motivation (exp13/exp14 lesson): instructions only help if they are RETRIEVED,
and abstract process-instructions do not embed near concrete data questions.
User queries, however, are full of API tokens (``np.isin``, ``.groupby``,
``plt.plot``) and object nouns. This generator therefore anchors every
instruction to a concrete API:

  1. census — scan the corpus code (documents + the ``examples`` table of the
     SQLite DB when ``path_to_db`` is given) with regexes and count API usages
     per library: qualified calls (``np.argsort``, ``pd.concat``) and method
     calls (``.groupby(``, ``.reshape(``);
  2. selection — top APIs by corpus frequency, quotas proportional to each
     library's share of the census;
  3. generation — ONE instruction document per API. The prompt forces the API
     token into the title (a how-do-I question about the practical task the
     API solves) and at least twice into the body, so the document carries the
     same lexical anchors a user query would; body = recommended modern usage
     + the common mistake + a tiny code example ending with ``result = ...``.

Same validation/packaging/audit as RagGuideGenerator (AST check, length
window, dedup, jsonl dump, yield counters).

NB: no ``from __future__ import annotations`` — keep annotations concrete for
the registry's dependency introspection.
"""

import logging
import re
import sqlite3
from collections import Counter
from typing import List, Optional, Tuple

from src.agent_constructor.core import Document
from src.generation.rag_guides import RagGuideGenerator

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


class ApiGuideGenerator(RagGuideGenerator):
    """Generate one instruction document per frequent corpus API.

    Args:
        url: base URL of the OpenAI-compatible LLM endpoint.
        model_name: model id to call.
        path_to_db: optional SQLite path — its ``examples`` table joins the
            census (that is where most of the corpus code lives).
        n_api_docs: total number of API instruction documents to generate.
        min_count: ignore APIs seen fewer than this many times in the census.
        num_workers / temperature / limit / max_doc_chars / dump_path: as in
            RagGuideGenerator.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        path_to_db: str = "",
        n_api_docs: int = 150,
        min_count: int = 5,
        num_workers: int = 8,
        temperature: float = 0.7,
        limit: int = 0,
        max_doc_chars: int = 1100,
        dump_path: str = "",
        name: str = "api_guide_generator",
    ):
        super().__init__(
            url=url, model_name=model_name, policy="v2",
            num_workers=num_workers, temperature=temperature, limit=limit,
            max_doc_chars=max_doc_chars, dump_path=dump_path, name=name,
        )
        self.path_to_db = path_to_db
        self.n_api_docs = int(n_api_docs)
        self.min_count = max(1, int(min_count))

    # ------------------------------------------------------------------ census
    def _census(self, documents: List[Document]) -> Counter:
        """Count (library, api_token) usages across corpus docs + examples."""
        counts: Counter = Counter()

        def scan(text: str, lib: Optional[str]) -> None:
            for m in _QUAL_RE.finditer(text):
                token = m.group(1)
                root = token.split(".")[0]
                counts[(_PREFIX_LIB.get(root, root), token)] += 1
            if lib:
                for m in _METH_RE.finditer(text):
                    meth = m.group(1)
                    if meth not in _METH_STOP:
                        counts[(lib, "." + meth)] += 1

        for d in documents:
            scan(d.text, (d.metadata or {}).get("library"))

        if self.path_to_db:
            try:
                con = sqlite3.connect(self.path_to_db)
                rows = con.execute(
                    "SELECT e.content, l.name FROM examples e "
                    "JOIN documents d ON e.doc_id = d.id "
                    "JOIN sections s ON d.section_id = s.id "
                    "JOIN libraries l ON s.library_id = l.id").fetchall()
                con.close()
                for content, lib in rows:
                    scan(content or "", lib)
                self.stats["census_examples"] = len(rows)
            except Exception as exc:
                logger.warning("examples census failed (%s): %s",
                               self.path_to_db, exc)
        return counts

    def _select_apis(self, counts: Counter) -> List[Tuple[str, str, int]]:
        """Top APIs overall, quotas proportional to library census share."""
        counts = Counter({k: v for k, v in counts.items()
                          if v >= self.min_count})
        if not counts:
            return []
        lib_totals = Counter()
        for (lib, _api), c in counts.items():
            lib_totals[lib] += c
        grand = sum(lib_totals.values())
        picked: List[Tuple[str, str, int]] = []
        for lib in sorted(lib_totals):
            quota = max(3, round(self.n_api_docs * lib_totals[lib] / grand))
            lib_apis = sorted(
                ((api, c) for (l, api), c in counts.items() if l == lib),
                key=lambda x: -x[1])
            # a bare method duplicating the tail of a qualified call of the
            # same library (".randn" vs "torch.randn") is the same API — keep
            # the qualified spelling only
            qual_tails = {a.rsplit(".", 1)[-1] for a, _ in lib_apis
                          if not a.startswith(".")}
            lib_apis = [(a, c) for a, c in lib_apis
                        if not (a.startswith(".") and a[1:] in qual_tails)]
            picked.extend((lib, api, c) for api, c in lib_apis[:quota])
        picked.sort(key=lambda x: -x[2])
        return picked[: self.n_api_docs]

    # -------------------------------------------------------------------- jobs
    def _api_jobs(self, apis: List[Tuple[str, str, int]]) -> List[Tuple[str, str, str]]:
        jobs = []
        for lib, api, _cnt in apis:
            shown = (f"the {lib} method `{api}()`" if api.startswith(".")
                     else f"`{api}`")
            jobs.append((lib, "api", (
                f"Write ONE short instruction document about using {shown} "
                f"correctly in modern {lib} code.\n"
                "Requirements:\n"
                f"- the TITLE is a how-do-I question a user would ask about "
                f"the practical task {shown} solves, and it MUST contain the "
                f"token {api};\n"
                f"- the body (2-4 sentences) gives the recommended modern "
                f"usage and the most common mistake with {shown}; mention the "
                f"token {api} at least twice in the text;\n"
                "- then a minimal code example (2-8 lines) with your own tiny "
                "invented data, ending with an assignment `result = ...`;\n"
                "- under 120 words plus the code.\n"
                "Format STRICTLY as:\nTITLE: <question containing the API "
                "token>\n<text, code in ``` fences>"
            )))
        return jobs

    # -------------------------------------------------------------- interface
    def generate(self, documents: List[Document]) -> List[Document]:
        self.stats = Counter()
        counts = self._census(documents)
        apis = self._select_apis(counts)
        if not apis:
            logger.warning("ApiGuideGenerator: empty API census")
            return []
        self.stats["census_apis"] = len(counts)
        logger.info("ApiGuideGenerator: census %d distinct APIs, selected %d "
                    "(top: %s)", len(counts), len(apis),
                    [f"{l}:{a}({c})" for l, a, c in apis[:8]])

        jobs = self._api_jobs(apis)
        if self.limit:
            jobs = jobs[: self.limit]
        parsed = self._run_jobs(jobs)
        return self._package(parsed, documents)
