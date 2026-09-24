"""PromptRuleGenerator — the rule block of the solver's system prompt, written by an LLM.

The generated text is delivered through the solver's ``system_prompt`` rather
than through the corpus: it is instruction, not documentation, and a retriever
has no query to match it against.

What lives where:

* **This module holds the knowledge** — ``PROBLEMS``: recurring ways an answer
  goes wrong, each written as a problem (symptom / why the check breaks / what
  to do instead), never as a finished rule.
* **The model holds the wording.** Every line of the final block is written by
  the LLM from those descriptions. No rule text is stored in the repository, so
  the block cannot be a copy of a hand-written one.

Two problem sets, selected by ``problem_set``:

``general``    the situations stated as properties of the task class — code
               completed inside an already-executed namespace, checked by
               execution on inputs the checker supplies. No benchmark
               artefacts, no harness markers, no variable names taken from it.
``specific``   the same 17 situations spelled out with the concrete artefacts
               they appear as.

Two writing modes, selected by ``mode``:

``per_problem``  one call per problem, the accepted rules carried into the next
                 call with two demands — do not repeat, do not contradict. A
                 problem an accepted rule already covers comes back as
                 ``COVERED: <n>`` and is skipped. This is the only
                 de-duplication: no embeddings, no clustering.
``batch``        one call carrying every problem; the model merges related ones
                 itself and returns at most ``n_rules`` rules, each tagged with
                 the problem ids it covers.

Every call — prompt, answer, verdict — is appended to ``rules_path`` (jsonl),
closing with the assembled text. With ``rebuild: False`` an existing file is
read back instead of calling the model, so a re-run evaluates the same bytes.

NB: no ``from __future__ import annotations`` — the pipeline registry
introspects concrete annotations (CLAUDE.md gotcha #1).
"""

import json
import logging
import os
import re

from typing import Dict, List, Optional

from openai import OpenAI

from src.agent_constructor.core import Block
from src.agent_constructor.prompts import PromptDiscoverable

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
_COVERED_RE = re.compile(r"^\s*COVERED\b\s*[:\-]?\s*(.*)$", re.I)
_RULE_LINE_RE = re.compile(
    r"^\s*RULE\s*:\s*(?P<rule>.+?)\s*(?:\|\s*COVERS\s*:\s*(?P<covers>.*))?$", re.I)
_LEAD_RE = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)])\s*")
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$")


# ---------------------------------------------------------------------------
# The knowledge. Descriptions of what goes wrong — never a finished rule.
# Both sets carry the same 17 keys, so one run can be read against the other.
# ---------------------------------------------------------------------------

PROBLEMS_GENERAL = [
    {
        "key": "repeat_imports",
        "symptom": (
            "The answer opens by restating the import lines already visible in "
            "the preamble above the gap, sometimes reordering them or adding "
            "imports for modules it never uses. Only after that does it get to "
            "the work that was actually requested."),
        "why": (
            "The preamble has already been executed before the answer is "
            "inserted, so those names are bound. Importing them again is at "
            "best dead weight in a short answer, and at worst rebinds a name "
            "the preamble deliberately pointed somewhere else."),
        "fix": (
            "Begin at the first line that does new work and rely on the names "
            "the preamble already bound."),
    },
    {
        "key": "recreate_inputs",
        "symptom": (
            "The answer re-declares the input objects it was given, filling "
            "them with the literal values shown in the illustration — the same "
            "numbers, the same table, the same text."),
        "why": (
            "The completion is executed against inputs the checker supplies, "
            "which are not the ones shown. Re-declaring the inputs throws the "
            "supplied data away, so the answer computes a correct result for "
            "the wrong problem and every hidden case fails."),
        "fix": (
            "Treat the inputs as already existing under the names the preamble "
            "gave them, and read from them rather than rebuilding them."),
    },
    {
        "key": "reseed_random",
        "symptom": (
            "The answer draws the random data again, or sets the seed once "
            "more before using the values, as if it had to reproduce the "
            "sample it can see."),
        "why": (
            "Drawing again replaces the objects under examination with a "
            "different sample, and reseeding shifts the sequence the rest of "
            "the code depends on. Either way the state inspected afterwards is "
            "not the state that was set up."),
        "fix": (
            "Use the sample that already exists and leave the generator's "
            "state alone."),
    },
    {
        "key": "placeholder_loader",
        "symptom": (
            "The preamble contains a call that reads like a data-loading "
            "helper, standing in for whatever produced the inputs, and the "
            "answer calls it again to obtain its own copy of them."),
        "why": (
            "No such helper exists at run time — it is a stand-in for data "
            "that has already been prepared. Calling it aborts the answer with "
            "an undefined-name error before anything is computed."),
        "fix": (
            "Take the objects it appears to produce as already present and "
            "work from them directly."),
    },
    {
        "key": "wrong_output_name",
        "symptom": (
            "The statement names the variable in which the answer is expected, "
            "and the answer stores its value somewhere else, or ends on a bare "
            "expression whose value is simply discarded."),
        "why": (
            "The check reads that one name out of the namespace after the "
            "answer has run. A value under any other name, or never bound at "
            "all, is indistinguishable from no answer."),
        "fix": (
            "Assign the final value to exactly the name the statement asks "
            "for, spelled the way it is spelled there."),
    },
    {
        "key": "print_instead_of_assign",
        "symptom": (
            "The answer computes the right thing and then prints it, or "
            "formats it into a message, instead of binding it to a name."),
        "why": (
            "Console output is not part of the state inspected afterwards. "
            "Whatever was printed is gone by the time the check looks for the "
            "value."),
        "fix": (
            "Bind the value to the requested name; print nothing."),
    },
    {
        "key": "missing_outputs",
        "symptom": (
            "Several values are requested — a pair, a triple, a value together "
            "with its accompanying mask or position — and the answer produces "
            "only the first of them and stops."),
        "why": (
            "Each requested value is inspected separately, so one missing "
            "binding fails the whole answer even when everything computed is "
            "correct."),
        "fix": (
            "Bind every value that was asked for, each under the name given "
            "for it."),
    },
    {
        "key": "inplace_target",
        "symptom": (
            "The statement asks for an existing object to end up changed, and "
            "the answer leaves the original untouched while putting the "
            "modified version into a freshly named variable."),
        "why": (
            "The object named in the statement is the one inspected. If the "
            "change lives elsewhere, the inspected object still holds its "
            "original contents."),
        "fix": (
            "Make the named object hold the changed state, whether by "
            "modifying it or by assigning the new version back to its name."),
    },
    {
        "key": "function_body_only",
        "symptom": (
            "The visible code ends on a function header whose body is missing, "
            "and the answer writes the header again, or writes a complete "
            "script at the left margin, or defines the function and then calls "
            "it and prints what came back."),
        "why": (
            "The answer is inserted directly underneath that header, so a "
            "repeated header nests a second definition, unindented code ends "
            "the definition with an empty body, and a trailing call runs work "
            "that was not asked for."),
        "fix": (
            "Write only the indented body, ending by returning the value the "
            "header is meant to produce."),
    },
    {
        "key": "wrappers",
        "symptom": (
            "The answer arrives wrapped: fenced code blocks, tag-style "
            "delimiters, a sentence of explanation before or after it, or a "
            "closing note about what the code does."),
        "why": (
            "Everything returned is treated as source text and executed as it "
            "stands. A fence marker or a sentence of prose is a syntax error "
            "on the first line, so nothing runs at all."),
        "fix": (
            "Return executable source and nothing else."),
    },
    {
        "key": "hardcoded_sizes",
        "symptom": (
            "Lengths, counts, positions and field names are written as the "
            "literals visible in the illustration — a loop bounded by the "
            "number of rows it happens to show, a position picked by eye, a "
            "field addressed by a name copied from the display."),
        "why": (
            "The checker runs the same answer on inputs of a different size "
            "and shape, where those literals point at the wrong place or out "
            "of bounds altogether."),
        "fix": (
            "Derive sizes, positions and names from the objects themselves at "
            "run time."),
    },
    {
        "key": "gratuitous_index_reset",
        "symptom": (
            "After computing the requested value, the answer rebuilds the row "
            "labels of the tabular result, so the labels become a plain "
            "running count and the original ones survive only as an extra "
            "column, if at all."),
        "why": (
            "The comparison covers the labels as well as the values. "
            "Renumbered labels, or an extra column carrying the old ones, "
            "differ from what is expected even when every value is right."),
        "fix": (
            "Leave the labelling as the operation produced it unless "
            "relabelling was part of the request."),
    },
    {
        "key": "unwrapped_container",
        "symptom": (
            "A labelled tabular or columnar object is requested, and the answer "
            "hands back the bare numeric buffer underneath it, stripping the "
            "labels on the way out."),
        "why": (
            "The comparison is made between objects of the expected type; a "
            "raw buffer has neither the labels nor the type that is checked "
            "for, so it is rejected before its contents are examined."),
        "fix": (
            "Return the object in the form that was asked for, with its "
            "labelling intact."),
    },
    {
        "key": "cosmetic_operations",
        "symptom": (
            "The answer adds finishing touches nobody asked for: converting to "
            "a tidier type, sorting the output, rounding the numbers, renaming "
            "fields to look neater."),
        "why": (
            "The comparison is exact. A changed type, a changed order or a "
            "rounded value differs from the expected object however much nicer "
            "it looks."),
        "fix": (
            "Perform the requested transformation and stop there."),
    },
    {
        "key": "plotting_context",
        "symptom": (
            "The drawing surface and its axes have already been created above "
            "the gap, and the answer starts a new one, or invents its own data "
            "to draw, or ends by calling for the picture to be displayed."),
        "why": (
            "The object inspected afterwards is the one the preamble created. "
            "Starting another leaves it empty, and a display call blocks or "
            "discards the figure depending on where it runs."),
        "fix": (
            "Add the requested elements to the surface and axes that already "
            "exist, under the names the preamble gave them, and end there."),
    },
    {
        "key": "submodule_import_style",
        "symptom": (
            "A function from a nested part of a scientific package is called "
            "in a spelling the preamble never established — the bare name when "
            "only the package root was imported, or a dotted path through a "
            "submodule that was never brought in."),
        "why": (
            "Importing a package root does not bind its submodules, and a name "
            "imported directly is not reachable through a dotted path. Either "
            "mismatch ends the answer with an attribute or name error."),
        "fix": (
            "Call the function through exactly the path the preamble's imports "
            "make available."),
    },
    {
        "key": "framework_era",
        "symptom": (
            "Code for a deep-learning framework is written in the style of its "
            "previous major version: a graph assembled first, symbolic input "
            "stand-ins declared, the computation then run inside an explicit "
            "execution context."),
        "why": (
            "The installed version evaluates operations as they are written "
            "and no longer provides those constructs, so the answer fails on "
            "the first of them."),
        "fix": (
            "Write straight-line code that computes and returns values "
            "immediately, in the framework's current style."),
    },
]


PROBLEMS_SPECIFIC = [
    {
        "key": "repeat_imports",
        "symptom": (
            "The answer starts with the same `import numpy as np` / `import "
            "pandas as pd` lines already written in the snippet above the gap, "
            "occasionally adding imports it never uses."),
        "why": (
            "The snippet's setup has already executed, so the modules are "
            "bound. Repeating the imports wastes the answer's first lines and "
            "can rebind an alias the setup pointed elsewhere."),
        "fix": (
            "Continue the snippet from the first line that does new work."),
    },
    {
        "key": "recreate_inputs",
        "symptom": (
            "The answer re-creates the input variables — `a`, `df`, `x`, `y`, "
            "`X`, `t` — assigning them the literal values printed in the "
            "example, e.g. `df = pd.DataFrame({'A': [1, 2, 3]})`."),
        "why": (
            "The completion is executed against hidden inputs bound to those "
            "same names. Re-creating them destroys the hidden data, so the "
            "answer solves the example instead of the task."),
        "fix": (
            "Use the input variables as they already are and never assign to "
            "them."),
    },
    {
        "key": "reseed_random",
        "symptom": (
            "The answer calls the random generator again — `np.random.rand(...)`, "
            "`np.random.randn(...)` — or repeats `np.random.seed(...)` before "
            "using the data."),
        "why": (
            "A second draw replaces the array under test, and a repeated seed "
            "shifts the stream the rest of the snippet relies on. The state "
            "that gets checked is no longer the state that was set up."),
        "fix": (
            "Work with the array the setup already drew and leave the seed "
            "untouched."),
    },
    {
        "key": "placeholder_loader",
        "symptom": (
            "The snippet shows `data = load_data()` and the answer calls "
            "`load_data()` again to get its own copy of the inputs."),
        "why": (
            "`load_data()` is a placeholder, not a real function: nothing "
            "defines it at run time. The call raises NameError and the answer "
            "dies before computing anything. The variables it appears to "
            "produce already exist."),
        "fix": (
            "Take the variables as given and never call the placeholder."),
    },
    {
        "key": "wrong_output_name",
        "symptom": (
            "The task asks for the answer in `result`, and the code leaves it "
            "in `answer`, `out`, `res` or `df2`, or ends on a bare expression "
            "whose value is dropped."),
        "why": (
            "The checker reads exactly the variable the task names out of the "
            "namespace afterwards. A value under another name, or never "
            "assigned, counts as no answer."),
        "fix": (
            "Assign the final value to precisely the variable name the task "
            "asks for."),
    },
    {
        "key": "print_instead_of_assign",
        "symptom": (
            "The answer computes the right value and ends with `print(result)` "
            "or `print(df.head())` instead of an assignment."),
        "why": (
            "Console output is not part of the namespace the checker inspects; "
            "the printed value is gone by the time the check runs."),
        "fix": (
            "Assign the value to the requested variable and print nothing."),
    },
    {
        "key": "missing_outputs",
        "symptom": (
            "The task asks for two or more outputs — a value and its index, a "
            "pair of arrays, a tuple — and the answer assigns only the first "
            "one."),
        "why": (
            "Every requested variable is checked separately, so one missing "
            "assignment fails the task even when the computation is right."),
        "fix": (
            "Assign all requested outputs, each under the name the task gives "
            "it."),
    },
    {
        "key": "inplace_target",
        "symptom": (
            "The task asks to modify `df` and the answer writes "
            "`df2 = df.drop(...)` or `result = df.rename(...)`, leaving `df` "
            "as it was."),
        "why": (
            "The checker inspects `df`. If the modification lives in another "
            "variable, `df` still holds its original contents."),
        "fix": (
            "Leave the result in `df`, either by modifying it in place or by "
            "assigning the new object back to `df`."),
    },
    {
        "key": "function_body_only",
        "symptom": (
            "The snippet ends on an unfinished `def f(df):` and the answer "
            "repeats the `def` line, or writes unindented code at column zero, "
            "or defines the function and then calls it: `f(df)`."),
        "why": (
            "The answer is inserted directly under that header. A repeated "
            "header nests a second definition, unindented code closes the "
            "header with an empty body, and a trailing call executes work that "
            "was not requested."),
        "fix": (
            "Write only the indented body plus its `return`."),
    },
    {
        "key": "wrappers",
        "symptom": (
            "The answer comes wrapped in ```python fences, in XML-style tags, "
            "or with a sentence of explanation before or after the code."),
        "why": (
            "Everything returned is executed as Python source. A fence marker "
            "or a line of prose is a syntax error on the first line, so "
            "nothing runs."),
        "fix": (
            "Emit raw Python source only."),
    },
    {
        "key": "hardcoded_sizes",
        "symptom": (
            "Sizes and positions are copied from the example as literals: "
            "`for i in range(5)`, `df.iloc[:3]`, `df['A']` where the column "
            "name came from the printed frame."),
        "why": (
            "The hidden inputs have different sizes and different column "
            "names, so the literals index the wrong elements or go out of "
            "bounds."),
        "fix": (
            "Derive sizes and names at run time — `len(a)`, `a.shape`, "
            "`df.columns`."),
    },
    {
        "key": "gratuitous_index_reset",
        "symptom": (
            "The answer finishes a grouping, filtering or sorting step and "
            "appends `.reset_index()` or `.reset_index(drop=True)` that the "
            "task never asked for."),
        "why": (
            "The comparison includes the index. Renumbering it, or spilling "
            "the old index into an extra column, differs from the expected "
            "frame even when the values match."),
        "fix": (
            "Leave the index exactly as the operation produced it."),
    },
    {
        "key": "unwrapped_container",
        "symptom": (
            "A DataFrame or Series is expected and the answer returns "
            "`df.values`, `series.to_numpy()` or `np.array(df)`."),
        "why": (
            "The check compares against a labelled pandas object. A bare "
            "ndarray has neither the index nor the type it looks for and is "
            "rejected before its contents are compared."),
        "fix": (
            "Return the DataFrame or Series itself, with its index and column "
            "labels intact."),
    },
    {
        "key": "cosmetic_operations",
        "symptom": (
            "The answer adds unrequested polish: `.astype(int)`, "
            "`.sort_values(...)`, `round(..., 2)`, renaming columns to look "
            "tidier."),
        "why": (
            "The comparison is exact — a changed dtype, a changed row order or "
            "a rounded value does not equal the expected object."),
        "fix": (
            "Do exactly what the task asks and nothing beyond it."),
    },
    {
        "key": "plotting_context",
        "symptom": (
            "The setup has already created the figure and axes, and the answer "
            "calls `plt.figure()` again, invents its own data to plot, or ends "
            "with `plt.show()`."),
        "why": (
            "The figure inspected afterwards is the one the setup created. A "
            "new figure leaves it empty, and `plt.show()` discards or blocks "
            "the rendering."),
        "fix": (
            "Add the requested drawing calls to the existing figure and axes, "
            "using the names the setup defined."),
    },
    {
        "key": "submodule_import_style",
        "symptom": (
            "The setup does `import scipy` and the answer calls a bare "
            "`solve_ivp(...)`, or the setup does `from scipy.integrate import "
            "solve_ivp` and the answer calls `scipy.integrate.solve_ivp(...)`."),
        "why": (
            "Importing `scipy` does not bind its submodules, and a directly "
            "imported name is not reachable through the dotted path. Either "
            "mismatch raises AttributeError or NameError."),
        "fix": (
            "Call the function through exactly the path the setup's imports "
            "establish."),
    },
    {
        "key": "framework_era",
        "symptom": (
            "TensorFlow code is written in the 1.x style: "
            "`tf.placeholder(...)`, a graph built first, then "
            "`with tf.Session() as sess: sess.run(...)`."),
        "why": (
            "The installed TensorFlow is 2.x and evaluates eagerly; those "
            "constructs no longer exist, so the answer fails on the first of "
            "them."),
        "fix": (
            "Write TF2 eager code that computes and returns tensors directly."),
    },
]


PROBLEM_SETS: Dict[str, List[dict]] = {
    "general": PROBLEMS_GENERAL,
    "specific": PROBLEMS_SPECIFIC,
}

# Label above the numbered list. Not a rule — every rule below it is generated.
_BLOCK_HEADER = "Follow these rules:"


def _strip_thinking(text: str) -> str:
    """Remove a reasoning preamble if the model emitted one anyway."""
    text = _THINK_RE.sub("", text or "")
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def _clean_rule(text: str) -> str:
    """One rule, at most two lines, stripped of list markers and wrappers."""
    lines = []
    for line in (text or "").splitlines():
        line = _FENCE_RE.sub("", line).strip()
        if not line:
            continue
        line = _LEAD_RE.sub("", line).strip().strip('"').strip()
        if line:
            lines.append(line)
        if len(lines) == 2:
            break
    return " ".join(lines).strip()


def _estimate_tokens(text: str) -> int:
    """Rough token count; no tokenizer, this only guards a budget warning."""
    return max(1, len(text or "") // 4)


class PromptRuleGenerator(PromptDiscoverable, Block):
    """Write the solver's rule block from the problem descriptions above.

    Args:
        url: base URL of the OpenAI-compatible endpoint that writes the rules.
        model_name: model id to request from that endpoint.
        mode: ``per_problem`` (one call per problem, accepted rules carried
            forward) or ``batch`` (one call, the model merges related problems).
        problem_set: ``general`` or ``specific`` — which description set the
            model is shown.
        n_rules: ceiling on the number of rules in ``batch`` mode.
        max_prompt_tokens: budget for the assembled block; exceeding it is
            logged, not enforced — silently dropping rules would be worse.
        temperature: sampling temperature for the writing calls.
        seed: request seed, recorded next to every call in the dump.
        max_tokens: ceiling per call.
        enable_thinking: sent as ``chat_template_kwargs``; False switches a
            reasoning model off, None sends nothing.
        rules_path: jsonl dump — every prompt, answer and verdict, then the
            assembled text.
        rebuild: True regenerates and overwrites the dump. False reuses the
            assembled text from an existing dump, so a re-run evaluates the
            same bytes.
        name: component name, also the prefix of its prompt keys.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        mode: str = "per_problem",
        problem_set: str = "general",
        n_rules: int = 12,
        max_prompt_tokens: int = 600,
        temperature: float = 0.3,
        seed: int = 7,
        max_tokens: int = 400,
        enable_thinking: Optional[bool] = False,
        rules_path: str = "",
        rebuild: bool = False,
        name: str = "prompt_rule_generator",
    ):
        super().__init__(name)

        if mode not in ("per_problem", "batch"):
            raise ValueError(f"unknown mode '{mode}': use per_problem or batch")
        if problem_set not in PROBLEM_SETS:
            raise ValueError(f"unknown problem_set '{problem_set}': "
                             f"use one of {sorted(PROBLEM_SETS)}")

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.mode = mode
        self.problem_set = problem_set
        self.problems = PROBLEM_SETS[problem_set]
        self.n_rules = int(n_rules)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.max_tokens = int(max_tokens)
        self.enable_thinking = enable_thinking
        self.rules_path = rules_path
        self.rebuild = bool(rebuild)

        self._rules = None
        self._records = []

    # ----------------------------------------------------------- interface
    def rules(self) -> str:
        """The assembled rule block, written on first use and then cached."""
        if self._rules is None:
            cached = self._load_cached()
            self._rules = cached if cached is not None else self._build()
        return self._rules

    # ----------------------------------------------------------------- LLM
    def _chat(self, user: str) -> str:
        extra = {}
        if self.enable_thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": self._system_prompt()},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            **({"extra_body": extra} if extra else {}),
        )
        return _strip_thinking(response.choices[0].message.content or "")

    # ------------------------------------------------------------- prompts
    def _system_prompt(self) -> str:
        return self.prompt("writer_system", (
            "You write the system prompt for a model whose job is to fill in a "
            "missing piece of code inside a snippet that has already been "
            "executed.\n\n"
            "How that model's answer is used:\n"
            "- the code above the gap has already run; its names are bound and "
            "its objects exist;\n"
            "- the answer is inserted at the gap and runs in that same "
            "namespace;\n"
            "- it is then checked by executing it on inputs the checker "
            "supplies, not the ones shown in the snippet, and by inspecting "
            "the values left behind.\n\n"
            "You are given descriptions of how answers go wrong. Turn a "
            "description into an instruction that prevents it.\n\n"
            "Every rule you write must be:\n"
            "- imperative, addressed to the answering model;\n"
            "- one or two lines, no longer;\n"
            "- free of any library, framework or product name, unless the "
            "description you were given is itself about one specific library;\n"
            "- free of any reference to a benchmark, a dataset, a grader or "
            "the markers such a harness uses;\n"
            "- bare: no heading, no numbering, no rationale, no commentary "
            "around it."))

    def _problem_block(self, problem: dict) -> str:
        return (f"PROBLEM ID: {problem['key']}\n"
                f"How the wrong answer looks:\n{problem['symptom']}\n"
                f"Why it fails the check:\n{problem['why']}\n"
                f"What the answering model should do instead:\n{problem['fix']}")

    def _per_problem_prompt(self, problem: dict, accepted: List[str]) -> str:
        if accepted:
            listing = "\n".join(f"{i + 1}. {rule}"
                                for i, rule in enumerate(accepted))
        else:
            listing = "(none yet — this is the first rule)"
        return self.render_prompt("per_problem_user", (
            "Here is one recurring problem.\n\n"
            "$problem\n\n"
            "Rules accepted so far:\n"
            "$accepted\n\n"
            "Write ONE rule that prevents this problem. It must neither repeat "
            "nor contradict any accepted rule above.\n"
            "If an accepted rule already prevents this problem, do not write a "
            "new one: answer with exactly 'COVERED: <number of that rule>' and "
            "nothing else.\n"
            "Otherwise answer with the rule text alone."),
            problem=self._problem_block(problem), accepted=listing)

    def _batch_prompt(self) -> str:
        problems = "\n\n".join(self._problem_block(p) for p in self.problems)
        return self.render_prompt("batch_user", (
            "Here are $n_problems recurring problems.\n\n"
            "$problems\n\n"
            "Write at most $n_rules rules covering all of them. Where several "
            "problems share one remedy, merge them into a single rule rather "
            "than writing near-duplicates; no two rules may overlap or "
            "contradict each other.\n"
            "Output one rule per line, in exactly this shape:\n"
            "RULE: <the rule> | COVERS: <problem ids it prevents, comma "
            "separated>\n"
            "Output nothing else — no heading, no numbering, no commentary."),
            problems=problems, n_problems=str(len(self.problems)),
            n_rules=str(self.n_rules))

    # ------------------------------------------------------------ building
    def _build(self) -> str:
        self._records = []
        logger.info(
            "%s: writing rules | mode=%s problems=%s (%d) model=%s seed=%d",
            self.name, self.mode, self.problem_set, len(self.problems),
            self.model_name, self.seed)

        rules = (self._build_batch() if self.mode == "batch"
                 else self._build_per_problem())

        if not rules:
            self._dump("", 0)
            raise ValueError(
                f"{self.name}: the model returned no usable rule — refusing to "
                f"run with an empty rule block (see {self.rules_path or 'logs'})")

        text = self._assemble(rules)
        estimate = _estimate_tokens(text)
        if estimate > self.max_prompt_tokens:
            logger.warning(
                "%s: the rule block is ~%d tokens, over the budget of %d — "
                "keeping it whole; trim the problem set if that matters",
                self.name, estimate, self.max_prompt_tokens)

        self._dump(text, estimate, n_rules=len(rules))
        logger.info("%s: %d rules, ~%d tokens", self.name, len(rules), estimate)
        return text

    def _build_per_problem(self) -> List[str]:
        accepted = []
        for problem in self.problems:
            prompt = self._per_problem_prompt(problem, accepted)
            answer = self._chat(prompt)
            covered = _COVERED_RE.match(answer.strip())

            if covered:
                reference = covered.group(1).strip() or "?"
                self._record(problem["key"], prompt, answer, "covered",
                             covers=[], covered_by=reference)
                logger.info("%s: %s covered by rule %s",
                            self.name, problem["key"], reference)
                continue

            rule = _clean_rule(answer)
            if not rule:
                self._record(problem["key"], prompt, answer, "rejected",
                             covers=[])
                logger.warning("%s: %s produced no usable rule",
                               self.name, problem["key"])
                continue

            accepted.append(rule)
            self._record(problem["key"], prompt, answer, "accepted",
                         covers=[problem["key"]], rule=rule)
        return accepted

    def _build_batch(self) -> List[str]:
        prompt = self._batch_prompt()
        answer = self._chat(prompt)

        rules = []
        claimed = set()
        for line in answer.splitlines():
            match = _RULE_LINE_RE.match(_FENCE_RE.sub("", line).strip())
            if not match:
                continue
            rule = _clean_rule(match.group("rule"))
            if not rule:
                continue
            covers = [token.strip() for token
                      in (match.group("covers") or "").split(",")
                      if token.strip()]
            rules.append(rule)
            claimed.update(covers)
            self._record(",".join(covers) or "?", prompt, line.strip(),
                         "accepted", covers=covers, rule=rule)
            if len(rules) >= self.n_rules:
                break

        # The whole answer as well, so a malformed tail stays visible in the audit.
        self._record("__raw__", prompt, answer,
                     "raw" if rules else "rejected", covers=[])
        if rules:
            missed = [p["key"] for p in self.problems if p["key"] not in claimed]
            if missed:
                logger.warning("%s: problems not claimed by any rule: %s",
                               self.name, ", ".join(missed))
        return rules

    def _assemble(self, rules: List[str]) -> str:
        listing = "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(rules))
        return f"{_BLOCK_HEADER}\n\n{listing}"

    # ----------------------------------------------------------------- dump
    def _record(self, key: str, prompt: str, response: str, status: str,
                covers: List[str], rule: str = "", covered_by: str = "") -> None:
        self._records.append({
            "record": "call",
            "key": key,
            "mode": self.mode,
            "problem_set": self.problem_set,
            "status": status,
            "covers": covers,
            "covered_by": covered_by,
            "rule": rule,
            "prompt": prompt,
            "response": response,
            "model": self.model_name,
            "seed": self.seed,
        })

    def _dump(self, text: str, estimate: int, n_rules: int = 0) -> None:
        if not self.rules_path:
            return
        os.makedirs(os.path.dirname(self.rules_path) or ".", exist_ok=True)
        with open(self.rules_path, "w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.write(json.dumps({
                "record": "final",
                "mode": self.mode,
                "problem_set": self.problem_set,
                "model": self.model_name,
                "seed": self.seed,
                "n_rules": n_rules,
                "est_tokens": estimate,
                "text": text,
            }, ensure_ascii=False) + "\n")

    def _load_cached(self):
        if not self.rules_path or self.rebuild:
            return None
        if not os.path.exists(self.rules_path):
            return None

        final = None
        with open(self.rules_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("record") == "final":
                    final = record

        if final is None or not final.get("text"):
            logger.warning("%s: %s carries no assembled text — regenerating",
                           self.name, self.rules_path)
            return None

        if (final.get("mode") != self.mode
                or final.get("problem_set") != self.problem_set):
            logger.warning(
                "%s: %s was written with mode=%s problem_set=%s while the "
                "config asks for mode=%s problem_set=%s — using the file as it "
                "stands; set rebuild: True to write it again",
                self.name, self.rules_path, final.get("mode"),
                final.get("problem_set"), self.mode, self.problem_set)

        logger.info("%s: reusing %d rules from %s (written by %s, seed %s)",
                    self.name, final.get("n_rules", 0), self.rules_path,
                    final.get("model"), final.get("seed"))
        return final["text"]
