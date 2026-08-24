"""Shared editorial contract for generated knowledge-base documents (exp17).

This module is the machine-checkable form of what actually moved the metric in
the manual corpus (exp12/exp15 analysis, Progress.md S1-S8). Both generators of
exp17 — ``PlanGuideGenerator`` and the repaired ``ApiGuideGenerator`` — build
their prompts from the fragments here and validate their output with
``check_contract``, so a comparison between them is a comparison of TOPIC
SELECTION and nothing else.

Two document classes survive the analysis:

``protocol``
    The only class with a strong measured effect (+11.6 pp within-library).
    States one rule about completing a partially written snippet: the setup has
    already run, the real inputs are hidden and differ from the shown example,
    the answer goes into the variable the task names. Declarative title, at most
    a few lines of code, no imports and no re-created data in the RECOMMENDED
    code.

``recipe``
    Question title, distinguishes two look-alike ways of doing a thing, short
    code on variables that already exist, anchored on a SPECIFIC (rare) API —
    corpus frequency of the APIs used by manual recipes has median 31 against
    398 for the exp15 generator.

Why imports / literal data are not banned outright: 18% of the manual protocol
docs and 17% of the manual recipes do contain them — inside an explicitly
marked WRONG example. The contract therefore bans them only in code that is not
introduced as a negative example.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DOC_CLASS_PROTOCOL = "protocol"
DOC_CLASS_RECIPE = "recipe"
DOC_CLASSES = (DOC_CLASS_PROTOCOL, DOC_CLASS_RECIPE)

# Variables a completion task is assumed to have already defined. Generated
# examples must reuse these instead of building their own data.
GIVEN_VARS = ("a", "arr", "df", "s", "x", "y", "X", "ax", "fig", "t", "model")

_CODE_LINE_RE = re.compile(r"^(?: {4,}|\t)\S")
_IMPORT_RE = re.compile(r"^\s*(?:import\s+[a-zA-Z_]|from\s+[\w.]+\s+import)")
_LITERAL_DATA_RE = re.compile(
    r"=\s*(?:pd\.DataFrame\s*\(|pd\.Series\s*\(\s*[\[{]|np\.array\s*\(\s*\[|"
    r"np\.random\.|torch\.tensor\s*\(\s*\[|torch\.randn\s*\(|torch\.rand\s*\(|"
    r"tf\.constant\s*\(\s*\[|\{\s*[\"'][A-Za-z_]\w*[\"']\s*:\s*\[)"
)
# Harness vocabulary of the benchmark harness must never appear (grant rule).
_HARNESS_RE = re.compile(
    r"BEGIN\s+SOLUTION|END\s+SOLUTION|SOLUTION\s+START|SOLUTION\s+END", re.I)
# A code block counts as a counter-example only when the line right above it is
# a short LEAD-IN saying so ("WRONG:", "If the task already contains lines
# like"). A long paragraph that merely mentions a common mistake and then shows
# the RECOMMENDED code does not qualify — that phrasing is exactly what the
# exp15 generator produced before every one of its examples.
_NEGATIVE_LEAD_RE = re.compile(
    r"\bwrong\b|\bbad\b|\bnever\b|\bdo not\b|\bdon't\b|\bharmful\b|"
    r"\balready (?:contains|shows|has|ran|run)\b|\binstead of\b|\bcopying\b|"
    r"\bif the (?:problem|task|snippet)\b", re.I)
_NEGATIVE_LEAD_MAX_CHARS = 90
# The two best-performing manual protocol docs (+17.2 and +14.5 pp) first quote
# the task's own setup — imports and literal data included — and only then show
# the one-line completion. That quoted setup is an illustration, not the
# recommendation, so the import/data rules must not apply to it.
_ILLUSTRATION_LEAD_RE = re.compile(
    r"\btask style\b|\btask text\b|\bthe task (?:shows|gives|ends|text)\b|"
    r"\bsetup code is given\b|\bproblem statement\b|\bsnippets? often\b|"
    r"\bfor example the task\b|\bshown before\b|\blines like\b|"
    r"\bsetups? like\b|\babbreviate\b|\bis shown\b", re.I)

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
_CONTRAST_RE = re.compile(
    r"instead of|rather than|\bvs\.?\b|difference between|confus|"
    r"wraps? around|silently|use .{0,40} when|but not\b|whereas", re.I)

# Angles for the protocol pack. Shared by every exp17 generator so the class is
# identical across methods and only recipe topics differ. The third field says
# whether a document from this angle must state one of the setup/hidden/assign
# facts — output-formatting angles legitimately do not.
PROTOCOL_ANGLES: Tuple[Tuple[str, str, bool], ...] = (
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


# --------------------------------------------------------------------- pieces
def code_blocks(content: str) -> List[Tuple[List[str], str]]:
    """Split a document into (code_lines, preceding_prose) blocks.

    Code is recognised the way ``RagGuideGenerator._parse_doc`` leaves it after
    unfencing: runs of indented lines.
    """
    lines = content.split("\n")
    blocks: List[Tuple[List[str], str]] = []
    i = 0
    while i < len(lines):
        if _CODE_LINE_RE.match(lines[i]):
            start = i
            while i < len(lines) and (_CODE_LINE_RE.match(lines[i]) or not lines[i].strip()):
                i += 1
            block = [l for l in lines[start:i] if l.strip()]
            prose_lines = [l.strip() for l in lines[max(0, start - 4):start]
                           if l.strip()]
            prose = prose_lines[-1] if prose_lines else ""
            blocks.append((block, prose))
        else:
            i += 1
    return blocks


def _is_illustrative(prose: str, block: List[str]) -> bool:
    """True when a block shows what NOT to do, or quotes the task's own setup.

    Either way it is not the recommended completion, so the no-import /
    no-invented-data rules do not apply to it.
    """
    short = len(prose) <= _NEGATIVE_LEAD_MAX_CHARS
    lead_ok = short and bool(_NEGATIVE_LEAD_RE.search(prose)
                             or _ILLUSTRATION_LEAD_RE.search(prose))
    head = block[0] if block else ""
    marked_inline = bool(re.search(r"#\s*(?:wrong|bad|never)", head, re.I))
    return lead_ok or marked_inline


def title_of(content: str) -> str:
    """First paragraph — manual titles routinely wrap onto a second line."""
    lines = []
    for line in content.split("\n"):
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line.strip())
    return " ".join(lines)


# ------------------------------------------------------------------ validator
def check_contract(content: str, doc_class: str, api_token: str = "",
                   require_facts: bool = True
                   ) -> Tuple[Optional[str], Dict[str, bool]]:
    """Validate one generated document against the contract.

    Returns ``(violation, soft_flags)``. ``violation`` is None when the document
    is acceptable, otherwise a short machine-readable reason. ``soft_flags``
    carries quality signals that are recorded but never reject a document.

    ``require_facts`` applies to the protocol class only: angles about output
    formatting (raw code, nothing extra) legitimately state no fact from the
    setup/hidden/assign trio, so their jobs switch the check off.
    """
    title = title_of(content)
    blocks = code_blocks(content)
    good_blocks = [b for b, prose in blocks if not _is_illustrative(prose, b)]
    good_lines = sum(len(b) for b in good_blocks)
    all_lines = sum(len(b) for b, _ in blocks)
    good_text = "\n".join("\n".join(b) for b in good_blocks)

    soft = {
        "has_contrast": bool(_CONTRAST_RE.search(content)),
        "has_wrong_example": len(blocks) != len(good_blocks),
        "reuses_given_var": any(
            re.search(r"\b%s\b" % re.escape(v), good_text) for v in GIVEN_VARS),
    }

    if _HARNESS_RE.search(content):
        return "harness_marker", soft
    if _IMPORT_RE.search(good_text):
        return "import_in_recommended_code", soft
    if _LITERAL_DATA_RE.search(good_text):
        return "invented_data_in_recommended_code", soft

    if doc_class == DOC_CLASS_PROTOCOL:
        if title.rstrip().endswith("?"):
            return "protocol_title_is_question", soft
        if good_lines > 3:
            return "protocol_code_too_long", soft
        if all_lines > 10:
            return "code_too_long", soft
        facts = sum(bool(rx.search(content)) for rx in _PROTOCOL_FACT_RES.values())
        soft["protocol_facts"] = facts >= 2
        if require_facts and facts < 1:
            return "protocol_facts_missing", soft
    elif doc_class == DOC_CLASS_RECIPE:
        if not title.rstrip().endswith("?"):
            return "recipe_title_not_question", soft
        if good_lines > 5:
            return "recipe_code_too_long", soft
        if all_lines > 10:
            return "code_too_long", soft
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
    common = (
        "- the code example uses variables that ALREADY EXIST in the reader's "
        "snippet (" + ", ".join(GIVEN_VARS[:8]) + "); it must NOT import "
        "anything and must NOT build its own DataFrame/array/tensor from "
        "literal values;\n"
        "- if you show a wrong way, put it in a separate block explicitly "
        "introduced with the word WRONG, and keep the recommended block clean;\n"
        "- no markdown headings, no bullet lists of unrelated tips, no closing "
        "summary;\n"
    )
    if doc_class == DOC_CLASS_PROTOCOL:
        return (
            "- the TITLE is a statement of the rule, NOT a question;\n"
            "- the body must state at least two of these three facts "
            "explicitly: the setup code has already run in the same namespace; "
            "the shown values are an illustration and the real inputs are "
            "hidden and different; the answer must be assigned to exactly the "
            "variable the task names;\n"
            "- at most 3 lines of recommended code;\n" + common)
    token = api_token or "the target API"
    return (
        "- the TITLE is a how-do-I question a user would actually type, and it "
        "contains " + token + ";\n"
        "- the body distinguishes TWO look-alike ways of doing this and says "
        "when each one is right — that distinction is the point of the "
        "document;\n"
        "- at most 5 lines of recommended code, sizes derived from the data "
        "rather than written as constants;\n" + common)


@dataclass
class GenJob:
    """One document to generate — the unit both exp17 generators work in."""

    library: str
    doc_class: str
    prompt: str
    api_token: str = ""
    require_facts: bool = True
    plan_id: str = ""
    topic: str = ""

    @property
    def prefix(self) -> str:
        return "proto" if self.doc_class == DOC_CLASS_PROTOCOL else "recipe"


def protocol_jobs(libs: List[str], n_docs: int = 0) -> List[GenJob]:
    """Build the protocol pack, deterministic in both order and content.

    One general document per angle first (they retrieve for every library),
    then per-library variants — the manual corpus shows both forms land in the
    top of the retrieval histogram (57 and 56 tasks for the two leaders).
    """
    jobs: List[GenJob] = []
    for key, angle, needs_facts in PROTOCOL_ANGLES:
        jobs.append(GenJob(
            library="general", doc_class=DOC_CLASS_PROTOCOL,
            prompt=_protocol_prompt("Python snippets", angle),
            require_facts=needs_facts, plan_id=f"proto:general:{key}",
            topic=key))
    for lib in sorted(libs):
        for key, angle, needs_facts in PROTOCOL_ANGLES:
            jobs.append(GenJob(
                library=lib, doc_class=DOC_CLASS_PROTOCOL,
                prompt=_protocol_prompt(f"{lib} snippets", angle),
                require_facts=needs_facts, plan_id=f"proto:{lib}:{key}",
                topic=key))
    return jobs[:n_docs] if n_docs else jobs


def _protocol_prompt(subject: str, angle: str) -> str:
    return (
        f"Write ONE short knowledge-base document for developers who complete "
        f"partially written {subject}.\n"
        f"The single rule it teaches: {angle}.\n"
        "Requirements:\n" + contract_rules(DOC_CLASS_PROTOCOL) +
        "- under 120 words plus the code.\n"
        "Format STRICTLY as:\nTITLE: <statement of the rule>\n"
        "<document text, code in ``` fences>"
    )
