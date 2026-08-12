"""Regression test for the RagGuideGenerator prompt refactor.

The prompts moved from bare literals to `self.prompt` / `self.render_prompt`
(src/agent_constructor/prompts.py) and from `{}` to `$` placeholders. This asserts
the rendered text is BYTE-IDENTICAL to what the pre-refactor code produced —
captured in test_rag_guide_prompts_golden.json — plus the mutation contract that
makes the prompts safe to optimize.

No network and no LLM: `openai` / `tqdm` are stubbed when absent, so this runs on
a laptop as well as on the server.

  python test_rag_guide_prompts.py
"""

import json
import os
import sys
import types

from typing import List


# --------------------------------------------------------------------------
# Import shims — the module imports openai/tqdm at module level, but nothing
# here calls a network. Real packages win when they are installed.
# --------------------------------------------------------------------------

_CHAT_CALLS: List[dict] = []


def _install_stubs():
    try:
        import openai  # noqa: F401
    except ImportError:
        module = types.ModuleType("openai")

        class _Completions:
            def create(self, **kwargs):
                _CHAT_CALLS.append(kwargs)
                message = types.SimpleNamespace(content="")
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        class _OpenAI:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=_Completions())

        module.OpenAI = _OpenAI
        sys.modules["openai"] = module

    try:
        import tqdm  # noqa: F401
    except ImportError:
        module = types.ModuleType("tqdm")
        module.tqdm = lambda iterable=None, **kwargs: (iterable or [])
        sys.modules["tqdm"] = module


_install_stubs()

from src.agent_constructor.core import Document                    # noqa: E402
from src.agent_constructor.prompts import Prompt                   # noqa: E402
from src.generation.rag_guides import RagGuideGenerator            # noqa: E402
from src.utils.prompt_registry import (                            # noqa: E402
    apply_candidate,
    collect_prompts,
    snapshot_prompts,
)


GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "test_rag_guide_prompts_golden.json")

_failures: List[str] = []


def chk(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        _failures.append(name)


def make_generator(policy, gold_args):
    """A generator whose knobs reproduce the golden capture exactly."""
    return RagGuideGenerator(
        url="http://stub/v1",
        model_name="stub-model",
        policy=policy,
        n_format_docs=gold_args["k"],          # one library -> per_lib == k
        n_migration_docs=2,                    # first 2 pandas checklist items
        n_recipe_docs=gold_args["docs_per_call"],
        seed=7,
    )


def corpus(gold_args):
    return [Document(id="1", text=gold_args["fragment"],
                     source="documents", metadata={"library": "pandas"})]


# --------------------------------------------------------------------------
# 1. Byte-identity with the pre-refactor prompts
# --------------------------------------------------------------------------

def test_goldens(gold):
    args = gold["_args"]
    for policy in ("v1", "v2"):
        gen = make_generator(policy, args)

        fmt = gen._format_jobs([args["lib"]])[0][2]
        chk(f"format_{policy} byte-identical", fmt == gold[f"format_{policy}"])

        migration = gen._migration_jobs([args["lib"]])[0][2]
        chk(f"migration_{policy} byte-identical",
            migration == gold[f"migration_{policy}"])

        recipe = gen._recipe_jobs(corpus(args))[0][2]
        chk(f"recipe_{policy} byte-identical", recipe == gold[f"recipe_{policy}"])

        gen._chat("ignored")
        system = _CHAT_CALLS[-1]["messages"][0]["content"]
        chk(f"chat_system byte-identical ({policy})", system == gold["chat_system"])


# --------------------------------------------------------------------------
# 2. Inventory completeness — variants register even when unused
# --------------------------------------------------------------------------

def test_inventory(gold):
    args = gold["_args"]
    gen = make_generator("v2", args)
    gen._format_jobs([args["lib"]])
    gen._migration_jobs([args["lib"]])
    gen._recipe_jobs(corpus(args))

    keys = {key for key, _ in gen.named_prompts()}
    expected = {
        "format_common", "format_v2_extra", "format_tail",
        "migration_style_v1", "migration_style_v2", "migration_tail",
        "recipe_main", "recipe_v2_rule", "recipe_tail",
    }
    chk("all prompt keys registered under v2", expected <= keys,
        f"missing={sorted(expected - keys)}")

    gen_v1 = make_generator("v1", args)
    gen_v1._format_jobs([args["lib"]])
    gen_v1._migration_jobs([args["lib"]])
    keys_v1 = {key for key, _ in gen_v1.named_prompts()}
    chk("v2-only variants still registered under v1",
        {"format_v2_extra", "migration_style_v2"} <= keys_v1,
        f"got={sorted(keys_v1)}")


# --------------------------------------------------------------------------
# 3. Mutation contract
# --------------------------------------------------------------------------

def test_mutation(gold):
    args = gold["_args"]
    gen = make_generator("v2", args)
    gen._format_jobs([args["lib"]])
    prompts = dict(gen.named_prompts())

    common = prompts["format_common"]
    before = common.fingerprint()
    common.set("Rewritten policy for $lib, $k docs.\n")
    rendered = gen._format_jobs([args["lib"]])[0][2]
    chk("set() reaches the component", rendered.startswith("Rewritten policy for pandas, 3 docs."))
    chk("fingerprint tracks mutation", common.fingerprint() != before)
    chk("revision bumped", common.revision == 1)

    common.rollback()
    chk("rollback restores text", common.fingerprint() == before)
    chk("rollback restores rendering",
        gen._format_jobs([args["lib"]])[0][2] == gold["format_v2"])

    try:
        common.set("policy that forgot the library placeholder, $k docs")
        chk("dropping $lib raises", False)
    except ValueError as exc:
        chk("dropping $lib raises", "lib" in str(exc))

    tail = prompts["format_tail"]
    try:
        tail.set("### SECTION\n")
        chk("frozen contract refuses set()", False)
    except ValueError as exc:
        chk("frozen contract refuses set()", "frozen" in str(exc))
    chk("frozen prompt marked optimizable=False", tail.optimizable is False)

    # temporarily() must restore even when the body raises
    baseline = common.text
    try:
        with common.temporarily("Temporary $lib / $k policy\n"):
            raise RuntimeError("evaluation blew up")
    except RuntimeError:
        pass
    chk("temporarily() restores after exception", common.text == baseline)


# --------------------------------------------------------------------------
# 4. Literal braces survive (the reason for string.Template)
# --------------------------------------------------------------------------

def test_braces():
    prompt = Prompt(
        'Return an empty dict {} for $lib and format with f\'{x:.2f}\'.',
        name="braces",
    )
    rendered = prompt.render(lib="pandas")
    chk("literal braces pass through",
        "{}" in rendered and "{x:.2f}" in rendered and "pandas" in rendered)

    try:
        Prompt("costs $5 today", name="bare-dollar")
        chk("bare '$' rejected", False)
    except ValueError as exc:
        chk("bare '$' rejected", "$$" in str(exc))


# --------------------------------------------------------------------------
# 5. Pipeline-level registry
# --------------------------------------------------------------------------

def test_registry(gold):
    args = gold["_args"]
    gen = make_generator("v2", args)
    gen._format_jobs([args["lib"]])

    pipeline = types.SimpleNamespace(_components={"generator": gen})
    found = collect_prompts(pipeline)
    chk("registry keys are component-qualified",
        "generator.format_common" in found, f"got={sorted(found)}")

    genome = snapshot_prompts(pipeline)
    apply_candidate(pipeline, {"generator.format_common": "New $lib / $k policy\n"})
    chk("apply_candidate mutates in place",
        found["generator.format_common"].text.startswith("New "))

    apply_candidate(pipeline, genome)
    chk("apply_candidate restores the genome",
        found["generator.format_common"].text == genome["generator.format_common"])

    try:
        apply_candidate(pipeline, {"generator.nope": "x"})
        chk("unknown key raises", False)
    except KeyError:
        chk("unknown key raises", True)


def main():
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold = json.load(f)

    print("=== goldens (byte-identity with pre-refactor prompts) ===")
    test_goldens(gold)
    print("\n=== inventory ===")
    test_inventory(gold)
    print("\n=== mutation contract ===")
    test_mutation(gold)
    print("\n=== literal braces ===")
    test_braces()
    print("\n=== pipeline registry ===")
    test_registry(gold)

    print()
    if _failures:
        print(f"{len(_failures)} CHECK(S) FAILED: {_failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
