"""Shared editorial contract for generated knowledge-base documents (exp18).

This module is the machine-checkable form of what the exp15/exp17 analysis
established about the manual corpus (Progress.md, S1-S8), plus the three
decisions taken after exp17:

* **no counter-examples at all.** exp17 documents carried 0.87-0.92 "WRONG"
  code blocks each against 0.01 in the manual recipes — because the exp17
  contract literally asked for them. The model is not frontier: two of those
  counter-examples turned out to call a function that does not exist
  (``scipy.special.ccdf``) or to state a false fact, so a counter-example is a
  hallucination risk that lands in the corpus as runnable code. A document now
  carries exactly ONE code block and it is the correct one; what not to do is a
  single sentence of prose.
* **minimum code, directive prose.** The class with the measured effect is the
  directive one (+11.6 pp against +1.0 pp for recipes), and it carries 0.8 lines
  of code. Thresholds below are calibrated on the manual profile.
* **form is shown, not described.** Rules stated in words were satisfied
  formally and missed in spirit twice; the prompts now carry exemplars taken
  from the manual corpus. Exemplars are FORM only — the loader marks them as
  such and a generated document whose topic matches an exemplar is rejected.

Two document classes:

``protocol``
    A conditional rule about completing a partially written snippet ("if the
    task shows X, do Y"). Declarative title, at most two lines of code, and it
    must name concrete variables or APIs — directives without a lexical anchor
    retrieve badly (2% of manual guide_ hits landed in both retrieval channels
    against 21% for recipes).

``recipe``
    Question title, distinguishes two look-alike ways of doing one thing, at
    most three lines of code on variables that already exist.
"""

import re
import sqlite3
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DOC_CLASS_PROTOCOL = "protocol"
DOC_CLASS_RECIPE = "recipe"
DOC_CLASSES = (DOC_CLASS_PROTOCOL, DOC_CLASS_RECIPE)

# Variables a completion task is assumed to have already defined. Generated
# examples must reuse these instead of building their own data.
GIVEN_VARS = ("a", "arr", "df", "s", "x", "y", "X", "ax", "fig", "t", "model")

# Form budget, calibrated on the 85th percentile of the manual corpus so the
# reference documents pass and only bloat is rejected. Measured there:
# guide_ code lines p50 1 / p85 3, prose p50 87 / p85 98;
# howto_ code lines p50 3 / p85 4, prose p50 50 / p85 60.
# The prompt asks for less than the cap — the cap only stops runaway drafts.
LIMITS = {
    DOC_CLASS_PROTOCOL: {"code_lines": 3, "prose_words": 100},
    DOC_CLASS_RECIPE: {"code_lines": 4, "prose_words": 70},
}
# One concrete name is already a lexical anchor: several manual recipes hang on
# a single API (plt.pie, scipy.spatial.distance.cdist) and they retrieve fine.
MIN_ANCHORS = 1

_CODE_LINE_RE = re.compile(r"^(?: {4,}|\t)\S")
_IMPORT_RE = re.compile(r"^\s*(?:import\s+[a-zA-Z_]|from\s+[\w.]+\s+import)", re.M)
_LITERAL_DATA_RE = re.compile(
    r"=\s*(?:pd\.DataFrame\s*\(|pd\.Series\s*\(\s*[\[{]|np\.array\s*\(\s*\[|"
    r"np\.random\.|torch\.tensor\s*\(\s*\[|torch\.randn\s*\(|torch\.rand\s*\(|"
    r"tf\.constant\s*\(\s*\[|\{\s*[\"'][A-Za-z_]\w*[\"']\s*:\s*\[)"
)
_HARNESS_RE = re.compile(
    r"BEGIN\s+SOLUTION|END\s+SOLUTION|SOLUTION\s+START|SOLUTION\s+END", re.I)
# Any dotted call counts as an anchor, not only the seven library prefixes:
# scaler.inverse_transform and optim.param_groups are just as searchable.
_API_TOKEN_RE = re.compile(
    r"\b[A-Za-z_]\w*\.[A-Za-z_]\w+(?:\.[A-Za-z_]\w+)*|"
    r"(?<![\w.])\.[a-z_]{3,}\b")
_WORD_RE = re.compile(r"[a-z_]{3,}")

# The three facts a protocol document exists to state.
_PROTOCOL_FACT_RES = {
    "setup_ran": re.compile(
        r"already (?:been )?(?:executed|run|ran|exists?|defined|filled)|"
        r"has already|runs? (?:right )?after|same namespace|"
        r"is inserted|continue the code|before your code", re.I),
    "hidden_values": re.compile(
        r"hidden|different (?:test |hidden )?(?:values|inputs|data)|"
        r"other inputs|grader|checker|may (?:substitute|differ|hold)|"
        r"real test data|actual input", re.I),
    "requested_var": re.compile(
        r"requested variable|variable the task|exact(?:ly)? the variable|"
        r"variable name the task|assign(?:ed|s|ing)? .{0,40}\b(?:result|"
        r"requested)\b|name the task (?:asks|requests|names)", re.I),
}
# Angles that teach something other than the setup/hidden/assign trio still
# have to state one thing explicitly — this is what the validator looks for.
_ANGLE_REQUIREMENT_RES = {
    "defbody": re.compile(r"indent|body|return\b", re.I),
    "onlyasked": re.compile(r"only what|nothing (?:extra|more)|unrequested|"
                            r"do not add|exactly the requested", re.I),
    "rawcode": re.compile(r"raw code|no (?:markdown|fences|tags|prose)|"
                          r"executable code only", re.I),
}

# ---------------------------------------------------------------- angle sets
# Six general angles, then six per library. exp17 reused one angle text across
# seven libraries and produced 2-7 near-duplicate pairs per corpus; these are
# distinct rules, not the same rule with a library name swapped.
GENERAL_ANGLES: Tuple[Tuple[str, str, bool], ...] = (
    ("setup", "the setup code shown above the gap has already been executed in "
              "the same namespace: the answer continues it and must not repeat "
              "imports, re-create the shown variables or call placeholder "
              "loader functions again", True),
    ("hidden", "the values printed in the task are only an illustration — the "
               "same completion is checked against hidden inputs of different "
               "size and content, so sizes must be derived from the data "
               "(len(a), a.shape, df.columns) and never hard-coded", True),
    ("assign", "the final value must end up in exactly the variable the task "
               "names (often result): a bare expression, a print() or a "
               "differently named variable counts as no answer", True),
    ("defbody", "when the task shows an unfinished function stub such as "
                "def f(data = example):, the answer is only the indented body "
                "ending with return — no new def header, no column-0 code, no "
                "call to the function afterwards", True),
    ("onlyasked", "do exactly what is asked: no unrequested reset_index, "
                  "astype, sorting, rounding or printing, and return the object "
                  "of exactly the requested type, name and shape", False),
    ("rawcode", "produce raw executable Python only — no markdown fences, no "
                "XML-ish tags, no prose before or after the code", False),
)

LIBRARY_ANGLES: Dict[str, Tuple[Tuple[str, str, bool], ...]] = {
    "pandas": (
        ("index", "an operation must leave the index the task expects: do not "
                  "call reset_index unless it was asked for, and keep the "
                  "original labels when the checker compares them", False),
        ("dtype", "return the object type the task names — a DataFrame stays a "
                  "DataFrame, a Series stays a Series, and .values or "
                  ".to_numpy() turns a correct answer into a wrong one", False),
        ("columns", "column names must match the task exactly, including "
                    "plural and singular spelling and their order", False),
        ("inplace", "assign the result back instead of relying on inplace "
                    "arguments, which pandas has been removing", False),
        ("groupselect", "when the task names the grouping key and the value "
                        "column, use exactly those and nothing else", False),
        ("dfvar", "operate on the DataFrame the snippet already defines "
                  "(usually df) — never rebuild it from the printed table", True),
    ),
    "numpy": (
        ("shape", "derive sizes from a.shape and len(a) rather than from the "
                  "numbers printed in the task", True),
        ("axis", "state the axis explicitly when the task talks about rows or "
                 "columns; the default axis is rarely the one meant", False),
        ("copyview", "an in-place operation on a slice changes the original "
                     "array — copy first when the task keeps the input", False),
        ("dtypecast", "keep the dtype the task implies; integer division and "
                      "float comparison silently change the answer", False),
        ("arrvar", "use the array the snippet already defines (a, arr) and do "
                   "not re-create it from the shown values", True),
        ("outvar", "put the array you produce into the exact name the task "
                   "asks for, even when it is not result", True),
    ),
    "matplotlib": (
        ("figopen", "the figure and axes already exist when your code runs: add "
                    "to them instead of creating a new figure", True),
        ("noshow", "do not call plt.show() — the checker inspects the current "
                   "figure's properties directly", False),
        ("axobj", "use the exact axis object the snippet defines (ax, axes, f) "
                  "rather than the pyplot state machine", True),
        ("nodata", "do not regenerate the plotted data: random values would "
                   "differ from the ones already drawn", True),
        ("labels", "set only the property the task names — a title is not a "
                   "suptitle and a legend label is not a tick label", False),
        ("style", "change style through the artist the task mentions instead "
                  "of global rcParams", False),
    ),
    "torch": (
        ("tensorvar", "the tensors already exist in the namespace with hidden "
                      "values — never rebuild them with torch.rand or from the "
                      "printed numbers", True),
        ("dtypeidx", "index tensors must be long and masks must be bool; "
                     "passing the wrong dtype raises instead of computing", False),
        ("shapederive", "take sizes from t.shape or t.size(0) rather than from "
                        "constants in the task text", True),
        ("nograd", "do not wrap the answer in training boilerplate: no "
                   "optimiser, no loop, no .backward() unless asked", False),
        ("device", "keep the tensor on the device it already lives on", False),
        ("outvar", "assign the resulting tensor to the variable the task "
                   "names", True),
    ),
    "sklearn": (
        ("preloaded", "X, y and the estimator are already prepared; "
                      "load_data() is a placeholder and must not be called", True),
        ("nofit", "refit only when the task asks — a pre-built model is "
                  "usually meant to be used as it is", False),
        ("shape2d", "estimators expect two-dimensional X: reshape a single "
                    "feature instead of passing a flat array", False),
        ("outvar", "store the prediction or the transformed data under the "
                   "exact name the task uses (predict, transformed, ...)", True),
        ("defaults", "pass only the parameters the task mentions and leave the "
                     "rest at their defaults", False),
        ("nosplit", "do not invent a train/test split the snippet does not "
                    "define", True),
    ),
    "scipy": (
        ("importstyle", "call the function through the module the snippet "
                        "imported: a bare name that was never imported raises "
                        "NameError", True),
        ("arrayin", "pass the arrays the snippet already defines instead of "
                    "rebuilding them", True),
        ("outvar", "assign the numeric result to the requested name without "
                   "printing or plotting it", True),
        ("params", "state the parameter the task names explicitly rather than "
                   "relying on the default", False),
        ("shapeout", "return the shape the task describes — many scipy calls "
                     "return a tuple where only one element is wanted", False),
        ("nostat", "do not add extra statistics the task did not request", False),
    ),
    "tensorflow": (
        ("eager", "write modern eager TF2 code: no Session, no placeholders, "
                  "no graph boilerplate", False),
        ("tensorvar", "the tensors already exist — do not re-create them with "
                      "tf.random or from the printed values", True),
        ("shapederive", "take sizes from tf.shape or .shape instead of "
                        "constants", True),
        ("outvar", "assign the resulting tensor to the name the task asks "
                   "for", True),
        ("nokeras", "do not build a model when the task asks for a tensor "
                    "operation", False),
        ("dtypemix", "match dtypes before combining tensors; TF refuses to mix "
                     "them silently", False),
    ),
}

# ------------------------------------------------------------------ exemplars
# Documents from the manual corpus used as FORM exemplars. Chosen among the ones
# that pass this contract; the measured leaders that quote the task setup as a
# second code block are deliberately not here — that shape is what we stopped
# generating.
EXEMPLAR_NAMES = {
    # Selected mechanically, not by taste: each one passes this contract, none
    # of them builds data from literals (an exemplar must not demonstrate what
    # the rules forbid), and their code operates on variables the reader is
    # assumed to already have. The first protocol entry is the measured leader
    # of its class (+14.5 pp on 57 tasks).
    DOC_CLASS_PROTOCOL: [
        "guide_numpy_snippet_completion",
        "guide_pandas_snippet_completion",
        "guide_sklearn_snippet_completion",
        "guide_load_data_placeholder",
        "guide_matplotlib_snippet_completion",
    ],
    DOC_CLASS_RECIPE: [
        "howto_inverse_permutation",
        "howto_argsort_top_n_descending",
        "howto_argmax_raveled_vs_unraveled",
        "howto_pad_array",
        "howto_one_hot_arbitrary_values",
    ],
}
_EXEMPLAR_CACHE: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}


def load_exemplars(db_path: str, doc_class: str) -> List[Tuple[str, str]]:
    """Fetch the exemplar pool for a class as [(doc_name, content), ...]."""
    key = (db_path, doc_class)
    if key in _EXEMPLAR_CACHE:
        return _EXEMPLAR_CACHE[key]
    names = EXEMPLAR_NAMES.get(doc_class, [])
    rows: List[Tuple[str, str]] = []
    if db_path and names:
        try:
            con = sqlite3.connect(db_path)
            placeholders = ",".join("?" * len(names))
            found = dict(con.execute(
                f"SELECT name, content FROM documents WHERE name IN ({placeholders})",
                names).fetchall())
            con.close()
            rows = [(n, found[n]) for n in names if n in found]
        except Exception:
            rows = []
    _EXEMPLAR_CACHE[key] = rows
    return rows


def exemplars_block(db_path: str, doc_class: str, rotation: int = 0,
                    n: int = 3) -> str:
    """Render `n` exemplars, rotated so the same topics do not recur everywhere."""
    pool = load_exemplars(db_path, doc_class)
    if not pool or n <= 0:
        return ""
    picked = [pool[(rotation + i) % len(pool)] for i in range(min(n, len(pool)))]
    parts = ["Study these documents. Copy their SHAPE — the length, the tone, "
             "the way one short code block answers the question. Do NOT reuse "
             "their subject: your document must be about the topic given above."]
    for i, (_name, content) in enumerate(picked, 1):
        parts.append(f"--- example {i} ---\n{content.strip()}")
    parts.append("--- end of examples ---")
    return "\n\n".join(parts)


def exemplar_topics(db_path: str, doc_class: str) -> List[str]:
    """Topic keys of the exemplars, for the copied-exemplar check."""
    return [topic_key(title_of(c)) for _n, c in load_exemplars(db_path, doc_class)]


# --------------------------------------------------------------------- pieces
def code_blocks(content: str) -> List[List[str]]:
    """Runs of indented lines — the shape ``_parse_doc`` leaves after unfencing."""
    lines = content.split("\n")
    blocks: List[List[str]] = []
    i = 0
    while i < len(lines):
        if _CODE_LINE_RE.match(lines[i]):
            start = i
            while i < len(lines) and (_CODE_LINE_RE.match(lines[i])
                                      or not lines[i].strip()):
                i += 1
            blocks.append([l for l in lines[start:i] if l.strip()])
        else:
            i += 1
    return blocks


def title_of(content: str) -> str:
    """First paragraph — titles routinely wrap onto a second line."""
    lines = []
    for line in content.split("\n"):
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line.strip())
    return " ".join(lines)


def topic_key(text: str) -> str:
    return " ".join(sorted(set(_WORD_RE.findall(text.lower()))))


def prose_of(content: str) -> str:
    """Everything that is not a code block."""
    return "\n".join(l for l in content.split("\n") if not _CODE_LINE_RE.match(l))


def strip_imports(content: str) -> Tuple[str, int]:
    """Drop import lines from code blocks instead of rejecting the document."""
    out, dropped = [], 0
    for line in content.split("\n"):
        if _CODE_LINE_RE.match(line) and _IMPORT_RE.match(line.strip()):
            dropped += 1
            continue
        out.append(line)
    return "\n".join(out), dropped


def keep_first_code_block(content: str) -> Tuple[str, int]:
    """Keep the first code block; drop later ones and the line introducing them.

    A second block is almost always the "wrong way" the model added out of
    habit; dropping it is cheaper than throwing the whole document away.
    """
    lines = content.split("\n")
    spans: List[Tuple[int, int]] = []          # (start, end) of each code block
    i = 0
    while i < len(lines):
        if _CODE_LINE_RE.match(lines[i]):
            start = i
            while i < len(lines) and (_CODE_LINE_RE.match(lines[i])
                                      or not lines[i].strip()):
                i += 1
            spans.append((start, i))
        else:
            i += 1
    if len(spans) < 2:
        return content, 0

    drop = set()
    removed = 0
    for start, end in spans[1:]:
        removed += sum(1 for l in lines[start:end] if l.strip())
        drop.update(range(start, end))
        # the lead-in sentence right above the dropped block goes with it
        j = start - 1
        while j >= 0 and not lines[j].strip():
            drop.add(j)
            j -= 1
        if j >= 0 and lines[j].strip() and j not in drop:
            lead = lines[j].strip()
            if len(lead) <= 90 and lead.endswith(":"):
                drop.add(j)
    kept = [l for k, l in enumerate(lines) if k not in drop]
    return "\n".join(kept).rstrip() + "\n", removed


# ------------------------------------------------------------------ validator
def check_contract(content: str, doc_class: str, api_token: str = "",
                   angle: str = "", exemplar_topics_: Optional[List[str]] = None
                   ) -> Tuple[Optional[str], Dict[str, bool]]:
    """Validate one generated document. Returns (violation | None, soft flags)."""
    title = title_of(content)
    blocks = code_blocks(content)
    code_lines = sum(len(b) for b in blocks)
    code_text = "\n".join("\n".join(b) for b in blocks)
    prose = prose_of(content)
    prose_words = len(re.findall(r"[A-Za-z][A-Za-z_]{1,}", prose))
    anchors = set(_API_TOKEN_RE.findall(content))
    anchors |= {v for v in GIVEN_VARS
                if re.search(r"(?<![\w])%s(?![\w])" % re.escape(v), code_text)}
    limits = LIMITS.get(doc_class, LIMITS[DOC_CLASS_RECIPE])

    soft = {
        "reuses_given_var": any(
            re.search(r"(?<![\w])%s(?![\w])" % re.escape(v), code_text)
            for v in GIVEN_VARS),
        "conditional_phrasing": bool(re.search(r"\bif\b.{0,80}\b(?:use|write|"
                                               r"assign|keep|call|do)\b",
                                               prose, re.I)),
    }

    if _HARNESS_RE.search(content):
        return "harness_marker", soft
    if len(blocks) > 1:
        return "more_than_one_code_block", soft
    if _IMPORT_RE.search(content):
        return "import_present", soft
    if _LITERAL_DATA_RE.search(content):
        return "invented_data", soft
    if code_lines > limits["code_lines"]:
        return "code_too_long", soft
    if prose_words > limits["prose_words"]:
        return "prose_too_long", soft
    if len(anchors) < MIN_ANCHORS:
        return "no_lexical_anchor", soft

    if exemplar_topics_:
        key = set(topic_key(title).split())
        for ex in exemplar_topics_:
            ex_set = set(ex.split())
            if ex_set and len(key & ex_set) / max(1, len(key | ex_set)) > 0.5:
                return "copied_exemplar", soft

    if doc_class == DOC_CLASS_PROTOCOL:
        if title.rstrip().endswith("?"):
            return "protocol_title_is_question", soft
        facts = {k: bool(rx.search(content)) for k, rx in _PROTOCOL_FACT_RES.items()}
        soft["protocol_facts"] = all(facts.values())
        if angle in ("setup", "hidden", "assign"):
            # These three angles exist to state the trio, so all three are due.
            missing = [k for k, ok in facts.items() if not ok]
            if missing:
                return "protocol_fact_missing:" + missing[0], soft
        elif angle in _ANGLE_REQUIREMENT_RES:
            if not _ANGLE_REQUIREMENT_RES[angle].search(content):
                return "angle_not_stated", soft
        # Library angles (pandas index, matplotlib figure, ...) teach their own
        # rule; demanding a fact from the trio there would only cost yield. The
        # general form rules above already apply to them.
    elif doc_class == DOC_CLASS_RECIPE:
        if not title.rstrip().endswith("?"):
            return "recipe_title_not_question", soft
        if api_token:
            bare = api_token.lstrip(".")
            if bare and bare.lower() not in content.lower():
                return "api_token_missing", soft
    else:
        return "unknown_doc_class", soft

    return None, soft


# -------------------------------------------------------------------- prompts
def contract_rules(doc_class: str, api_token: str = "") -> str:
    """The shared editorial rules, rendered for one document class."""
    limits = LIMITS.get(doc_class, LIMITS[DOC_CLASS_RECIPE])
    common = (
        f"- EXACTLY ONE code block, and it is the correct way. Never write a "
        f"wrong, bad or 'before' version as code — if something must be warned "
        f"about, say it in ONE sentence of prose;\n"
        f"- the code uses variables that ALREADY EXIST in the reader's snippet "
        f"({', '.join(GIVEN_VARS[:8])}). If you need an import, you are writing "
        f"the wrong kind of example: the reader's imports are already done;\n"
        f"- never build data from literal values (no pd.DataFrame({{...}}), no "
        f"np.array([...]), no torch.rand);\n"
        f"- keep it short: no more than {limits['code_lines']} lines of code "
        f"and {limits['prose_words']} words of prose, and prefer fewer;\n"
        f"- name concrete things: the variable the reader already has (df, arr, "
        f"ax, ...) and the API you recommend (np.argsort, df.groupby, ...). A "
        f"document that names nothing concrete is never found by search;\n"
        # A short checklist of "do not" items is exactly how the strongest
        # manual directives read, so bullets are allowed — what is banned is a
        # second code block and a closing restatement.
        f"- no markdown headings and no closing summary; a short list of "
        f"do-not items in prose is fine;\n"
    )
    if doc_class == DOC_CLASS_PROTOCOL:
        return (
            "- the TITLE states the rule, it is NOT a question;\n"
            "- write the rule in the directive form: if the task looks like "
            "THIS, then do THAT;\n"
            "- state explicitly: the setup code has already run in the same "
            "namespace; the shown values are an illustration and the real "
            "inputs are hidden and different; the answer goes into exactly the "
            "variable the task names;\n" + common)
    token = api_token or "the target API"
    return (
        f"- the TITLE is a how-do-I question a user would actually type, and it "
        f"contains {token};\n"
        "- the body distinguishes TWO look-alike ways of doing this and says "
        "when each one is right — that distinction is the whole point;\n"
        "- write it directively: if the task wants THIS, use THAT;\n"
        "- take sizes from the data, never from constants;\n" + common)


SELF_CHECK = (
    "Before answering, check your draft:\n"
    "1. exactly one code block, and it is the correct way?\n"
    "2. no import line and no invented data anywhere?\n"
    "3. does it name at least two concrete variables or APIs?\n"
    "4. is the title in the required form?\n"
    "If any answer is no, fix it before you reply.\n"
)


PROTOCOL_PROMPT = """\
You are writing one document for a knowledge base that is read by a developer
who is completing a partially written $subject. The setup code above the gap has
already run; the developer only needs the missing lines.

Purpose of this document: teach ONE rule so the reader stops making one specific
mistake. It is an instruction, not a tutorial.

The rule it teaches: $angle

Form:
$rules- under 120 words in total.

$examples
$selfcheck
Answer STRICTLY as:
TITLE: <the rule, as a statement>
<the document>
"""


@dataclass
class GenJob:
    """One document to generate — the unit both generators work in."""

    library: str
    doc_class: str
    prompt: str
    api_token: str = ""
    angle: str = ""
    plan_id: str = ""
    topic: str = ""

    @property
    def prefix(self) -> str:
        return "proto" if self.doc_class == DOC_CLASS_PROTOCOL else "recipe"


def angles_for(lib: str) -> Tuple[Tuple[str, str, bool], ...]:
    return LIBRARY_ANGLES.get(lib, ())


def protocol_jobs(libs: List[str], n_docs: int = 0, render=None,
                  fewshot_db_path: str = "", n_fewshot: int = 3) -> List[GenJob]:
    """Build the directive pack: six general rules plus six per library.

    ``render(key, default, **values)`` lets the caller register the template
    through its own prompt store, so the text lands in prompts.json.
    """
    if render is None:
        def render(_key, default, **values):
            return string.Template(default).substitute(**values)

    plan: List[Tuple[str, str, str, str]] = []      # lib, subject, key, angle
    for key, angle, _ in GENERAL_ANGLES:
        plan.append(("general", "Python snippet", key, angle))
    for lib in sorted(libs):
        for key, angle, _ in angles_for(lib):
            plan.append((lib, f"{lib} snippet", key, angle))

    jobs: List[GenJob] = []
    for i, (lib, subject, key, angle) in enumerate(plan):
        prompt = render(
            "protocol", PROTOCOL_PROMPT, subject=subject, angle=angle,
            rules=contract_rules(DOC_CLASS_PROTOCOL),
            examples=exemplars_block(fewshot_db_path, DOC_CLASS_PROTOCOL,
                                     rotation=i, n=n_fewshot),
            selfcheck=SELF_CHECK)
        jobs.append(GenJob(library=lib, doc_class=DOC_CLASS_PROTOCOL,
                           prompt=prompt, angle=key,
                           plan_id=f"proto:{lib}:{key}", topic=key))
    return jobs[:n_docs] if n_docs else jobs
