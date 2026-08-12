"""Candidates: the unit an optimizer proposes.

A candidate is a plain ``{prompt_name: text}`` mapping over the *optimizable*
prompts found in a built pipeline. Frozen prompts — the ``### DOC`` format
contracts — never enter a candidate, so an optimizer cannot break the parsing
its own score depends on.

Keys are ``Prompt.name`` (``"<component_name>.<prompt_key>"``, e.g.
``"rag_guide_generator.format_common"``). That is deliberately the same key
``prompts.set_overrides`` looks up, so ``best_prompts.json`` can be replayed by
``run_ds1000.py --prompts`` without any translation.

Applying a candidate mutates the ``Prompt`` objects in place, which is what
lets components pick up new text without the pipeline being rebuilt.
"""

import contextlib

from typing import Dict, Iterator, Tuple

from src.agent_constructor.prompts import Prompt
from src.utils.prompt_registry import collect_prompts


Candidate = Dict[str, str]
PromptMap = Dict[str, Prompt]


def from_pipeline(pipeline) -> PromptMap:
    """Every prompt the pipeline declares, keyed by ``Prompt.name``."""
    found: PromptMap = {}
    for prompt in collect_prompts(pipeline).values():
        found[prompt.name] = prompt
    return found


def optimizable(prompts: PromptMap) -> PromptMap:
    """The subset an optimizer is allowed to rewrite."""
    return {name: prompt for name, prompt in prompts.items() if prompt.optimizable}


def seed_candidate(prompts: PromptMap) -> Candidate:
    """Current text of every optimizable prompt — the search's start point."""
    return {name: prompt.text for name, prompt in optimizable(prompts).items()}


def apply(prompts: PromptMap, candidate: Candidate) -> None:
    """Write `candidate` into the prompt objects, in place.

    Unknown or frozen names raise: a candidate naming a prompt outside the
    search space means the optimizer and the pipeline disagree.
    """
    targets = optimizable(prompts)
    unknown = sorted(set(candidate) - set(targets))
    if unknown:
        raise KeyError(f"not optimizable prompts of this pipeline: {unknown}")
    for name, text in candidate.items():
        if targets[name].text != text:
            targets[name].set(text)


@contextlib.contextmanager
def applied(prompts: PromptMap, candidate: Candidate) -> Iterator[None]:
    """Apply `candidate` for the duration of the block, then restore."""
    previous = seed_candidate(prompts)
    apply(prompts, candidate)
    try:
        yield
    finally:
        apply(prompts, previous)


def describe(candidate: Candidate) -> Iterator[Tuple[str, int]]:
    for name in sorted(candidate):
        yield name, len(candidate[name])
