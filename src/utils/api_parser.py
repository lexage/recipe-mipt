from __future__ import annotations

import ast
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src.benchmarks.ds1000 import DataItemDS1000

# ── output dataclass ──────────────────────────────────────────────────────────

@dataclass
class APICall:
    qualified: str          # e.g. "pandas.DataFrame.groupby"
    count: int = 1


@dataclass
class ParseResult:
    problem_id: int | str
    library: str
    apis: list[APICall]
    unresolved: list[str]   # attribute/call names we could not route to a library


# ═══════════════════════════════════════════════════════════════════════════════
#  STATIC LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════════════════════

# Maps a root module alias to its canonical package name.
# Populated from import statements; we seed a few universal aliases here.
_UNIVERSAL_ALIASES: dict[str, str] = {
    "np":      "numpy",
    "pd":      "pandas",
    "plt":     "matplotlib.pyplot",
    "sns":     "seaborn",
    "tf":      "tensorflow",
    "torch":   "torch",
    "nn":      "torch.nn",
    "F":       "torch.nn.functional",
    "scipy":   "scipy",
    "sklearn": "sklearn",
    "sp":      "scipy",
}

# When a call like pd.DataFrame(...) / np.array(...) is seen we note that the
# result belongs to a specific *class* (for method-chain resolution).
# Key: (library_root, function_name)  →  class_tag
_PRODUCER_TABLE: dict[tuple[str, str], str] = {
    # pandas constructors / readers
    ("pandas", "DataFrame"):         "pandas.DataFrame",
    ("pandas", "Series"):            "pandas.Series",
    ("pandas", "read_csv"):          "pandas.DataFrame",
    ("pandas", "read_excel"):        "pandas.DataFrame",
    ("pandas", "read_json"):         "pandas.DataFrame",
    ("pandas", "read_html"):         "pandas.DataFrame",
    ("pandas", "concat"):            "pandas.DataFrame",
    ("pandas", "merge"):             "pandas.DataFrame",
    ("pandas", "pivot_table"):       "pandas.DataFrame",
    ("pandas", "get_dummies"):       "pandas.DataFrame",
    ("pandas", "cut"):               "pandas.Series",
    ("pandas", "qcut"):              "pandas.Series",
    ("pandas", "to_datetime"):       "pandas.Series",
    # numpy constructors
    ("numpy", "array"):              "numpy.ndarray",
    ("numpy", "zeros"):              "numpy.ndarray",
    ("numpy", "ones"):               "numpy.ndarray",
    ("numpy", "arange"):             "numpy.ndarray",
    ("numpy", "linspace"):           "numpy.ndarray",
    ("numpy", "random"):             "numpy",            # sub-namespace
    ("numpy", "empty"):              "numpy.ndarray",
    ("numpy", "full"):               "numpy.ndarray",
    ("numpy", "eye"):                "numpy.ndarray",
    ("numpy", "concatenate"):        "numpy.ndarray",
    ("numpy", "stack"):              "numpy.ndarray",
    ("numpy", "vstack"):             "numpy.ndarray",
    ("numpy", "hstack"):             "numpy.ndarray",
    # matplotlib – plt.gca/plt.figure return Axes/Figure
    ("matplotlib.pyplot", "gca"):    "matplotlib.axes.Axes",
    ("matplotlib.pyplot", "figure"): "matplotlib.figure.Figure",
    ("matplotlib.pyplot", "subplots"): "matplotlib.subplots_result",  # special-cased below
    # sklearn estimators – .fit() returns self, .predict() returns ndarray
    # (handled generically via class tag prefix)
    # tensorflow
    ("tensorflow", "Variable"):      "tensorflow.Variable",
    ("tensorflow", "constant"):      "tensorflow.Tensor",
    # torch
    ("torch", "tensor"):             "torch.Tensor",
    ("torch", "zeros"):              "torch.Tensor",
    ("torch", "ones"):               "torch.Tensor",
    ("torch", "randn"):              "torch.Tensor",
    ("torch", "rand"):               "torch.Tensor",
    ("torch", "arange"):             "torch.Tensor",
    ("torch", "from_numpy"):         "torch.Tensor",
}

# Pandas accessor namespaces:  df.str / df.dt / df.cat → sub-namespace class
_PANDAS_ACCESSORS: dict[str, str] = {
    "str":  "pandas.core.strings.StringMethods",
    "dt":   "pandas.core.indexes.accessors.DatetimeProperties",
    "cat":  "pandas.core.arrays.categorical.CategoricalAccessor",
    "sparse": "pandas.core.arrays.sparse.accessor.SparseAccessor",
}

# Pandas groupby return type
_PANDAS_GROUPBY_RETURNS = "pandas.core.groupby.DataFrameGroupBy"
_PANDAS_SERIES_GROUPBY_RETURNS = "pandas.core.groupby.SeriesGroupBy"

# Methods on DataFrame / Series that return the same class (for chaining)
_PANDAS_SELF_METHODS: set[str] = {
    "copy", "reset_index", "set_index", "rename", "drop", "fillna",
    "dropna", "sort_values", "sort_index", "assign", "astype",
    "replace", "clip", "abs", "round", "transpose", "T",
    "reindex", "pipe", "head", "tail", "sample", "query",
    "filter", "where", "mask", "interpolate", "ffill", "bfill",
    "explode", "melt", "stack", "unstack", "pivot", "pivot_table",
    "merge", "join", "append",
}

# Methods that return a Series from a DataFrame
_PANDAS_DF_TO_SERIES: set[str] = {
    "squeeze", "idxmax", "idxmin", "any", "all", "sum",
    "mean", "median", "std", "var", "min", "max", "count",
    "nunique", "value_counts", "cumsum", "cumprod",
    "diff", "pct_change", "rank",
}

# sklearn classes whose .fit() returns self and .predict()/.transform() return ndarray
_SKLEARN_ESTIMATOR_PREFIXES = ("sklearn.",)


# ── default variable name → class for each library ───────────────────────────

_LIBRARY_DEFAULTS: dict[str, dict[str, str]] = {
    "Pandas": {
        "df":  "pandas.DataFrame",
        "df1": "pandas.DataFrame",
        "df2": "pandas.DataFrame",
        "s":   "pandas.Series",
        "idx": "pandas.Index",
        "a":   "numpy.ndarray",
        "arr": "numpy.ndarray",
        "x":   "numpy.ndarray",
        "y":   "numpy.ndarray",
    },
    "Numpy": {
        "a":   "numpy.ndarray",
        "arr": "numpy.ndarray",
        "x":   "numpy.ndarray",
        "y":   "numpy.ndarray",
        "b":   "numpy.ndarray",
        "df":  "numpy.ndarray",
    },
    "Matplotlib": {
        "ax":  "matplotlib.axes.Axes",
        "ax1": "matplotlib.axes.Axes",
        "ax2": "matplotlib.axes.Axes",
        "fig": "matplotlib.figure.Figure",
        "x":   "numpy.ndarray",
        "y":   "numpy.ndarray",
        "df":  "pandas.DataFrame",
    },
    "Tensorflow": {
        "x": "tensorflow.Tensor",
        "y": "tensorflow.Tensor",
        "a": "tensorflow.Tensor",
        "b": "tensorflow.Tensor",
        "A": "tensorflow.Tensor",
        "B": "tensorflow.Tensor",
    },
    "Scipy": {
        "x":   "numpy.ndarray",
        "y":   "numpy.ndarray",
        "a":   "numpy.ndarray",
        "arr": "numpy.ndarray",
    },
    "Sklearn": {
        "X":       "numpy.ndarray",
        "y":       "numpy.ndarray",
        "X_train": "numpy.ndarray",
        "X_test":  "numpy.ndarray",
        "y_train": "numpy.ndarray",
        "y_test":  "numpy.ndarray",
        "df":      "pandas.DataFrame",
    },
    "Pytorch": {
        "x":   "torch.Tensor",
        "y":   "torch.Tensor",
        "a":   "torch.Tensor",
        "b":   "torch.Tensor",
        "inp": "torch.Tensor",
        "out": "torch.Tensor",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SYMBOL TABLE
# ═══════════════════════════════════════════════════════════════════════════════

class SymbolTable:
    """
    Maps local variable/alias names → class tags (strings like 'pandas.DataFrame').
    Two layers:
      • aliases: import alias → canonical module  (e.g. 'pd' → 'pandas')
      • vars:    variable name → class tag         (e.g. 'df' → 'pandas.DataFrame')
    """

    def __init__(self, library: str) -> None:
        self.aliases: dict[str, str] = dict(_UNIVERSAL_ALIASES)
        self.vars: dict[str, str] = dict(_LIBRARY_DEFAULTS.get(library, {}))

    # ── import recording ──────────────────────────────────────────────────────

    def record_import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.aliases[local] = alias.name

    def record_import_from(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            # e.g. 'from scipy import stats' → stats → scipy.stats
            self.aliases[local] = f"{module}.{alias.name}" if module else alias.name

    # ── alias resolution ──────────────────────────────────────────────────────

    def resolve_module(self, name: str) -> Optional[str]:
        """Return canonical module for an alias, or None."""
        return self.aliases.get(name)

    def resolve_var(self, name: str) -> Optional[str]:
        """Return class tag for a variable, or None."""
        return self.vars.get(name)

    def bind_var(self, name: str, class_tag: str) -> None:
        self.vars[name] = class_tag

    # ── dotted-attribute chain helpers ────────────────────────────────────────

    def dotted_chain(self, node: ast.expr) -> Optional[list[str]]:
        """
        Unwrap a chain of pure attribute accesses into a list of name parts.
        ast.Attribute(value=ast.Attribute(value=ast.Name('np'), attr='linalg'), attr='norm')
        → ['np', 'linalg', 'norm']
        Returns None if the chain contains a Call or Subscript node at the root
        (those are handled by visit_Call via _infer_receiver_class).
        """
        parts: list[str] = []
        n = node
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
            return list(reversed(parts))
        return None  # non-Name root (Call, Subscript, BinOp, Compare…)

    def infer_receiver_class(self, node: ast.expr) -> Optional[str]:
        """
        For nodes that are NOT a plain Name (i.e. they are Subscript, BinOp,
        Compare, or a chained Call), try to infer the class of the result so
        that subsequent method calls on it can be resolved.
        """
        # Subscript: df.iloc[List] → class of df.iloc result = pandas.DataFrame
        if isinstance(node, ast.Subscript):
            slc = node.slice

            # Helper: should a pandas DataFrame subscript downcast to Series?
            # Only when the key is definitely a single scalar column name —
            # i.e. a string constant.  Everything else (Name variable, List,
            # Slice, integer constant, …) is assumed to preserve the DataFrame.
            def _should_downcast(cls: Optional[str]) -> bool:
                if cls != "pandas.DataFrame":
                    return False
                return isinstance(slc, ast.Constant) and isinstance(slc.value, str)

            # GroupBy subscript: SeriesGroupBy["col"] → SeriesGroupBy (already Series-level)
            def _groupby_subscript(cls: Optional[str]) -> Optional[str]:
                if cls == "pandas.core.groupby.DataFrameGroupBy":
                    # gb["col"] → SeriesGroupBy
                    if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                        return "pandas.core.groupby.SeriesGroupBy"
                    # gb[["a","b"]] → DataFrameGroupBy still
                    return cls
                return None

            # 1. Pure dotted chain as value: df.iloc[…] / df.loc[…]
            chain = self.dotted_chain(node.value)
            if chain:
                _, ret_class = self.resolve_attr_chain_pub(chain)
                gb = _groupby_subscript(ret_class)
                if gb is not None:
                    return gb
                if _should_downcast(ret_class):
                    ret_class = "pandas.Series"
                return ret_class

            # 2. Plain Name: df[…] / result[…]
            if isinstance(node.value, ast.Name):
                cls = self.resolve_var(node.value.id)
                gb = _groupby_subscript(cls)
                if gb is not None:
                    return gb
                if _should_downcast(cls):
                    cls = "pandas.Series"
                return cls

            # 3. Call or Attribute as value: groupby(…)["b"] / mask(…)[::-1] / df[col].iloc[rows]
            #    Delegate to the extractor's _visit_and_get_class (set at runtime)
            if isinstance(node.value, (ast.Call, ast.Attribute)):
                get_cls = getattr(self, '_get_class_fn', None)
                if get_cls:
                    cls = get_cls(node.value)
                    gb = _groupby_subscript(cls)
                    if gb is not None:
                        return gb
                    if _should_downcast(cls):
                        cls = "pandas.Series"
                    return cls

            return None

        # BinOp / Compare: (df.a + df.b) or (df.a > 0)
        # Result is typically a pandas Series or numpy array — use primary library
        if isinstance(node, (ast.BinOp, ast.Compare)):
            # Try left operand
            if isinstance(node, ast.Compare):
                left = node.left
            else:
                left = node.left
            chain = self.dotted_chain(left)
            if chain:
                _, ret = self.resolve_attr_chain_pub(chain)
                return ret
            if isinstance(left, ast.Name):
                return self.resolve_var(left.id)
            return None

        # UnaryOp: ~mask, -arr
        if isinstance(node, ast.UnaryOp):
            chain = self.dotted_chain(node.operand)
            if chain:
                _, ret = self.resolve_attr_chain_pub(chain)
                return ret
            if isinstance(node.operand, ast.Name):
                return self.resolve_var(node.operand.id)
            return None

        return None

    def resolve_attr_chain_pub(self, chain: list[str]):
        """Stub — overridden at runtime by the extractor that owns this table."""
        return None, None

    def set_chain_resolver(self, fn) -> None:
        self.resolve_attr_chain_pub = fn


# ═══════════════════════════════════════════════════════════════════════════════
#  FIRST PASS: build symbol table from setup + reference imports
# ═══════════════════════════════════════════════════════════════════════════════

def _build_symbol_table(setup_src: str, ref_src: str, library: str) -> SymbolTable:
    st = SymbolTable(library)

    def _record_imports(tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                st.record_import(node)
            elif isinstance(node, ast.ImportFrom):
                st.record_import_from(node)

    def _record_assignments(tree: ast.AST) -> None:
        """
        Walk top-level assignments to bind variable names to class tags.
        e.g.  df = pd.DataFrame(...)  → df → pandas.DataFrame
              a  = np.array(...)      → a  → numpy.ndarray
        """
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            class_tag = _infer_class_from_call(node.value, st)
            if class_tag is None:
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    st.bind_var(tgt.id, class_tag)
                elif isinstance(tgt, ast.Tuple):
                    # e.g. fig, ax = plt.subplots()
                    # special-case: plt.subplots → (Figure, Axes)
                    if class_tag == "matplotlib.subplots_result":
                        _bind_subplots_tuple(tgt, st)
                    else:
                        # generic: bind each element with the same tag (best-effort)
                        for elt in tgt.elts:
                            if isinstance(elt, ast.Name):
                                st.bind_var(elt.id, class_tag)

    for src in (setup_src, ref_src):
        if not src.strip():
            continue
        tree = _safe_parse(src)
        if tree is None:
            continue
        _record_imports(tree)
        _record_assignments(tree)

    return st


def _bind_subplots_tuple(tgt: ast.Tuple, st: SymbolTable) -> None:
    """
    Handle:
      fig, ax        = plt.subplots()
      fig, (ax1,ax2) = plt.subplots(1, 2)
      fig, axes      = plt.subplots(...)
    """
    elts = tgt.elts
    if len(elts) >= 1 and isinstance(elts[0], ast.Name):
        st.bind_var(elts[0].id, "matplotlib.figure.Figure")
    if len(elts) >= 2:
        second = elts[1]
        if isinstance(second, ast.Name):
            st.bind_var(second.id, "matplotlib.axes.Axes")
        elif isinstance(second, ast.Tuple):
            for elt in second.elts:
                if isinstance(elt, ast.Name):
                    st.bind_var(elt.id, "matplotlib.axes.Axes")


def _infer_class_from_call(node: ast.expr, st: SymbolTable) -> Optional[str]:
    """Given the RHS of an assignment, return the class tag if we can determine it."""
    if not isinstance(node, ast.Call):
        return None
    chain = st.dotted_chain(node.func)
    if chain is None:
        return None
    root = chain[0]
    canonical_root = st.resolve_module(root) or st.resolve_var(root) or root
    # Try to match against producer table
    # e.g. chain = ['pd', 'DataFrame'] → root canonical 'pandas', fn 'DataFrame'
    # e.g. chain = ['plt', 'subplots']
    lib_root = canonical_root.split(".")[0]   # 'pandas', 'numpy', etc.
    fn_name = chain[-1]

    # Direct lookup (full canonical → fn)
    for key_root in (canonical_root, lib_root):
        tag = _PRODUCER_TABLE.get((key_root, fn_name))
        if tag:
            return tag

    # sklearn estimators: any sklearn.* class's constructor
    if canonical_root.startswith("sklearn."):
        return canonical_root  # e.g. 'sklearn.linear_model.LinearRegression'

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SAFE PARSE
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_parse(src: str) -> Optional[ast.AST]:
    """Try to parse src as a module; on failure, wrap it in a function."""
    for attempt in (
        src,
        "def _wrap():\n" + textwrap.indent(src, "    "),
        "if True:\n"     + textwrap.indent(src, "    "),
    ):
        try:
            return ast.parse(attempt)
        except SyntaxError:
            pass
    return None


def _extract_prompt_setup(prompt: str) -> str:
    blocks = re.findall(r"<code>(.*?)</code>", prompt, re.DOTALL)
    return blocks[-1].strip() if blocks else ""


# ═══════════════════════════════════════════════════════════════════════════════
#  SECOND PASS: emit API calls from reference_code AST
# ═══════════════════════════════════════════════════════════════════════════════

class APIExtractor(ast.NodeVisitor):
    """
    Walks an AST and records every API call / property access that can be
    attributed to one of the seven DS-1000 libraries.
    """

    def __init__(self, st: SymbolTable) -> None:
        self.st = st
        self._counts: dict[str, int] = defaultdict(int)
        self._unresolved: list[str] = []
        self._node_return: dict[int, Optional[str]] = {}
        self._lambda_receiver_stack: list[Optional[str]] = []
        # Wire back-references so SymbolTable helpers can call extractor methods
        self.st.set_chain_resolver(self._resolve_attr_chain)
        self.st._get_class_fn = self._visit_and_get_class

    # ── helpers ───────────────────────────────────────────────────────────────

    def _emit(self, qualified: str) -> None:
        self._counts[qualified] += 1

    def _emit_unresolved(self, name: str) -> None:
        self._unresolved.append(name)

    def _qualify(self, receiver_class: str, attr: str) -> str:
        return f"{receiver_class}.{attr}"

    def _module_for_alias(self, name: str) -> Optional[str]:
        return self.st.resolve_module(name)

    def _class_for_var(self, name: str) -> Optional[str]:
        return self.st.resolve_var(name)

    # ── resolve a dotted attribute node to (qualified_api, return_class) ──────

    def _resolve_attr_chain(
        self, chain: list[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Given ['df', 'groupby'] or ['np', 'linalg', 'norm'] or ['ax', 'set_xlabel'],
        return (qualified_api_string, return_class_tag_or_None).
        """
        if not chain:
            return None, None

        root = chain[0]

        # ── Case 1: root is a module alias  (np.array, plt.subplots, tf.constant) ──
        canonical = self._module_for_alias(root)
        if canonical:
            lib = canonical.split(".")[0]
            rest = chain[1:]
            if not rest:
                return canonical, canonical
            # Build the qualified name step by step
            qualified = f"{canonical}.{'.'.join(rest)}"
            # Try to get return type from producer table
            fn_name = rest[-1]
            return_class = _PRODUCER_TABLE.get((canonical, fn_name)) or \
                           _PRODUCER_TABLE.get((lib, fn_name))
            return qualified, return_class

        # ── Case 2: root is a known variable  (df.groupby, a.reshape, x.assign) ──
        receiver_class = self._class_for_var(root)
        if receiver_class:
            return self._resolve_method_chain(receiver_class, chain[1:])

        # ── Case 3: unknown root ──────────────────────────────────────────────
        return None, None

    def _resolve_method_chain(
        self, receiver_class: str, attrs: list[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Walk attrs left-to-right, emitting each method and updating receiver_class.
        Returns the (last qualified name, final receiver class) for the whole chain.
        """
        if not attrs:
            return receiver_class, receiver_class

        last_qualified = None
        for attr in attrs:
            qualified, receiver_class = self._resolve_one_method(receiver_class, attr)
            if qualified:
                last_qualified = qualified
            else:
                # Can't resolve further; emit what we have and stop
                break

        return last_qualified, receiver_class

    def _resolve_one_method(
        self, receiver_class: str, attr: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Given a receiver class tag and an attribute name, return
        (qualified_api, new_receiver_class_for_chaining).
        """
        lib = receiver_class.split(".")[0]

        # ── pandas ────────────────────────────────────────────────────────────
        if lib == "pandas":
            # Accessors: .str / .dt / .cat
            if attr in _PANDAS_ACCESSORS:
                accessor_class = _PANDAS_ACCESSORS[attr]
                return self._qualify(receiver_class, attr), accessor_class

            # groupby → DataFrameGroupBy or SeriesGroupBy
            if attr == "groupby":
                gb_class = (
                    _PANDAS_SERIES_GROUPBY_RETURNS
                    if "Series" in receiver_class
                    else _PANDAS_GROUPBY_RETURNS
                )
                return self._qualify(receiver_class, attr), gb_class

            # iloc / loc / at / iat — indexers, keep class
            if attr in ("iloc", "loc", "at", "iat"):
                return self._qualify(receiver_class, attr), receiver_class

            # self-returning methods
            if attr in _PANDAS_SELF_METHODS:
                return self._qualify(receiver_class, attr), receiver_class

            # → Series reduction
            if attr in _PANDAS_DF_TO_SERIES and "DataFrame" in receiver_class:
                return self._qualify(receiver_class, attr), "pandas.Series"

            # apply / map / agg (stay in same type for resolution purposes)
            if attr in ("apply", "map", "agg", "aggregate", "transform"):
                return self._qualify(receiver_class, attr), receiver_class

            # fallback: emit qualified, keep receiver
            return self._qualify(receiver_class, attr), receiver_class

        # ── numpy ─────────────────────────────────────────────────────────────
        if lib == "numpy":
            return self._qualify(receiver_class, attr), receiver_class

        # ── matplotlib ────────────────────────────────────────────────────────
        if lib == "matplotlib":
            return self._qualify(receiver_class, attr), receiver_class

        # ── tensorflow ────────────────────────────────────────────────────────
        if lib == "tensorflow":
            return self._qualify(receiver_class, attr), receiver_class

        # ── torch ─────────────────────────────────────────────────────────────
        if lib == "torch":
            return self._qualify(receiver_class, attr), receiver_class

        # ── sklearn estimators ────────────────────────────────────────────────
        if lib == "sklearn":
            if attr in ("fit", "fit_transform", "set_params", "partial_fit"):
                return self._qualify(receiver_class, attr), receiver_class  # returns self
            if attr in ("predict", "transform", "predict_proba", "score"):
                return self._qualify(receiver_class, attr), "numpy.ndarray"
            return self._qualify(receiver_class, attr), receiver_class

        # ── scipy ─────────────────────────────────────────────────────────────
        if lib == "scipy":
            return self._qualify(receiver_class, attr), receiver_class

        return None, receiver_class

    # ── visit_Call ────────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        """
        Emit the API for this call and store the return class in
        self._node_return[id(node)] for any parent call that needs it.
        """
        func     = node.func
        return_class: Optional[str] = None

        # ── A: pure dotted chain as func  →  np.linalg.norm(…) / df.copy(…) ──
        chain = self.st.dotted_chain(func)
        if chain:
            qualified, return_class = self._resolve_attr_chain(chain)
            if qualified:
                self._emit_chain_calls(chain, qualified)
            else:
                self._emit_unresolved(".".join(chain))

        # ── B: func is Attribute whose value is a Call/Subscript/BinOp ────────
        elif isinstance(func, ast.Attribute):
            attr      = func.attr
            inner     = func.value
            inner_cls = self._visit_and_get_class(inner)

            if inner_cls:
                qualified, return_class = self._resolve_one_method(inner_cls, attr)
                if qualified:
                    self._emit(qualified)
                else:
                    self._emit_unresolved(attr)
            else:
                self._emit_unresolved(attr)

        # ── C: bare name call or anything else ────────────────────────────────
        # nothing to emit; just fall through to recurse into args

        # ── record return class for parent ────────────────────────────────────
        self._node_return[id(node)] = return_class

        # ── recurse into arguments, passing lambda-receiver context ───────────
        self._lambda_receiver_stack.append(return_class)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)
        self._lambda_receiver_stack.pop()

    def _visit_and_get_class(self, node: ast.expr) -> Optional[str]:
        """
        Visit node (which may be a Call, Attribute, Subscript, BinOp, Compare,
        or Name) and return the class tag of its result.
        """
        if isinstance(node, ast.Call):
            self.visit(node)
            return self._node_return.get(id(node))

        if isinstance(node, ast.Name):
            return self.st.resolve_var(node.id)

        # Pure attribute chain with a non-Name root, e.g. df['datetime'].dt
        # dotted_chain returns None for these, so we resolve step by step.
        if isinstance(node, ast.Attribute):
            inner_cls = self._visit_and_get_class(node.value)
            if inner_cls:
                qualified, ret_cls = self._resolve_one_method(inner_cls, node.attr)
                if qualified:
                    self._emit(qualified)
                return ret_cls
            return None

        # Subscript / BinOp / Compare / UnaryOp
        cls = self.st.infer_receiver_class(node)
        self.visit(node)   # recurse for nested API calls inside
        return cls

    def _emit_chain_calls(self, chain: list[str], _final_qualified: str) -> None:
        """
        Emit each step in a dotted chain individually.
        e.g. ['df', 'groupby'] → emit pandas.DataFrame.groupby
             ['np', 'linalg', 'norm'] → emit numpy.linalg.norm
             ['plt', 'gca'] → emit matplotlib.pyplot.gca
        """
        if not chain:
            return
        root = chain[0]
        canonical = self.st.resolve_module(root)
        if canonical:
            # Module chain: emit the full dotted path as one API
            qualified = f"{canonical}.{'.'.join(chain[1:])}" if len(chain) > 1 else canonical
            self._emit(qualified)
            return

        # Variable chain: emit each method step
        receiver_class = self.st.resolve_var(root)
        if not receiver_class:
            return
        for attr in chain[1:]:
            qualified, receiver_class = self._resolve_one_method(receiver_class, attr)
            if qualified:
                self._emit(qualified)
            if not receiver_class:
                break

    # ── visit_Attribute (bare property access, not a call) ────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Emit bare attribute accesses that are not wrapped in a Call.
        e.g.  a.shape / df.columns / df.T
        We only emit these if the receiver is known.
        """
        # Skip if parent is a Call (visit_Call already handles it)
        # There's no easy parent reference in ast, so we rely on the fact
        # that visit_Call recurses *into* func nodes via dotted_chain which
        # won't re-visit (we don't call generic_visit for the func subtree).
        chain = self.st.dotted_chain(node)
        if chain and len(chain) >= 2:
            root = chain[0]
            if not self.st.resolve_module(root) and not self.st.resolve_var(root):
                # unresolvable
                pass
            else:
                qualified, _ = self._resolve_attr_chain(chain)
                if qualified:
                    self._emit(qualified)

        # Still recurse into the value for nested attrs
        self.generic_visit(node)

    # ── visit_Assign (update symbol table from assignments in ref code) ────────

    def visit_Assign(self, node: ast.Assign) -> None:
        # Try fast path first: direct constructor call (pd.DataFrame(), np.array(), …)
        class_tag = _infer_class_from_call(node.value, self.st)
        if class_tag:
            # Constructor recognised — bind targets, then recurse into RHS normally
            self._bind_targets(node.targets, class_tag)
            self.visit(node.value)
        else:
            # Slow path: visit the RHS (emitting API calls) and capture its return class
            class_tag = self._visit_and_get_class(node.value)
            if class_tag:
                self._bind_targets(node.targets, class_tag)

    def _bind_targets(self, targets, class_tag: str) -> None:
        """Bind a list of assignment targets to class_tag in the symbol table."""
        for tgt in targets:
            if isinstance(tgt, ast.Name):
                self.st.bind_var(tgt.id, class_tag)
            elif isinstance(tgt, ast.Tuple):
                if class_tag == "matplotlib.subplots_result":
                    _bind_subplots_tuple(tgt, self.st)
                else:
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            self.st.bind_var(elt.id, class_tag)

    # ── visit_Lambda (for lambdas inside .apply / .map / .agg) ───────────────

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """
        Bind lambda parameters to typed classes before recursing.
        The receiver context (class of the object whose .apply/.map/.agg
        received this lambda) is read from _lambda_receiver_stack.
        """
        receiver = (
            self._lambda_receiver_stack[-1]
            if self._lambda_receiver_stack else None
        )
        if receiver and receiver.startswith("pandas"):
            param_class = "pandas.Series"
        elif receiver and receiver.startswith("numpy"):
            param_class = "numpy.ndarray"
        elif receiver and receiver.startswith("torch"):
            param_class = "torch.Tensor"
        else:
            param_class = "pandas.Series"  # safe default for DS-1000

        saved: dict[str, Optional[str]] = {}
        for arg in node.args.args:
            saved[arg.arg] = self.st.resolve_var(arg.arg)
            self.st.bind_var(arg.arg, param_class)

        self.generic_visit(node)

        for name, prev in saved.items():
            if prev is None:
                self.st.vars.pop(name, None)
            else:
                self.st.bind_var(name, prev)

    # ── result extraction ─────────────────────────────────────────────────────

    # Python builtins that appear as unresolved but are not library APIs
    _PYTHON_BUILTINS: frozenset[str] = frozenset({
        "len", "range", "list", "tuple", "dict", "set", "int", "float", "str",
        "bool", "type", "zip", "enumerate", "map", "filter", "sorted", "reversed",
        "print", "repr", "abs", "round", "sum", "min", "max", "any", "all",
        "open", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
        "iter", "next", "super", "object", "staticmethod", "classmethod",
        "property", "vars", "dir", "id", "hash", "callable",
    })

    def results(self) -> tuple[list[APICall], list[str]]:
        apis = [APICall(qualified=k, count=v) for k, v in sorted(self._counts.items())]
        unresolved = sorted(
            name for name in set(self._unresolved)
            if name not in self._PYTHON_BUILTINS
            and not name.isidentifier() is False  # keep valid names only
        )
        return apis, unresolved


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_api_calls(item: DataItemDS1000) -> ParseResult:
    """
    Analyse a DS-1000 item and return the APIs used in its reference solution.

    Parameters
    ----------
    item : DataItemDS1000

    Returns
    -------
    ParseResult
        • problem_id  – from metadata
        • library     – primary library tag (e.g. 'Pandas')
        • apis        – list of APICall(qualified, count), sorted alphabetically
        • unresolved  – attribute chains we could not route to a known library
    """
    library   = item.metadata.get("library", "")
    setup_src = _extract_prompt_setup(item.prompt)
    ref_src   = item.reference_code

    # 1. Build symbol table (imports + assignments from setup AND reference)
    st = _build_symbol_table(setup_src, ref_src, library)

    # 2. Parse reference_code
    ref_tree = _safe_parse(ref_src)
    if ref_tree is None:
        return ParseResult(
            problem_id=item.p_id,
            library=library,
            apis=[],
            unresolved=[f"PARSE_ERROR: {ref_src[:80]}"],
        )

    # 3. Walk and extract
    extractor = APIExtractor(st)
    extractor.visit(ref_tree)
    apis, unresolved = extractor.results()

    return ParseResult(
        problem_id=item.p_id,
        library=library,
        apis=apis,
        unresolved=unresolved,
    )
