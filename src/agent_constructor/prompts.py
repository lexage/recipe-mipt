"""Prompts as discoverable, mutable objects — declared inline where they are used.

Motivation: every LLM-bearing component carries its prompt as an inline literal, so
a prompt change is a code change that leaves no trace in a results folder, and
nothing can enumerate or swap the prompts (needed for logging and for any prompt
optimizer).

The prompt text stays written in the method that uses it. Wrap the literal in
``self.prompt(key, default)``: the first call registers it into a per-instance
store, later calls return the live — possibly optimized — text.

    class MyAgent(PromptDiscoverable, Agent):
        def run(self, task):
            system = self.prompt("system", "You are a helpful assistant.")
            user = self.render_prompt("user", "Solve $task.", task=task)

Placeholders use ``string.Template`` syntax (``$name``), not ``str.format``: these
prompts routinely contain literal braces (``{"key": "value"}``, ``f'{x:.2f}'``) and
``str.format`` would crash on them. Write ``$$`` for a literal dollar sign.

NB: no ``from __future__ import annotations`` — the pipeline registry introspects
concrete annotations (CLAUDE.md gotcha #1).
"""

import contextlib
import hashlib
import string

from typing import Dict, FrozenSet, Iterator, List, Tuple


_OVERRIDES: Dict[str, str] = {}


def set_overrides(mapping: Dict[str, str]) -> None:
    """Process-wide prompt overrides, applied when a prompt registers.

    Registration is lazy and happens inside the method that uses the prompt, so
    there is no earlier hook to swap the text: by the time a component object
    exists, its prompts may not be registered yet, and by the time they are, a
    generator has already run. Overrides are consulted at registration instead.

    Keys are either ``"<component>.<key>"`` or the bare ``"<key>"``. The default
    text still defines the placeholder contract — an override that drops one is
    rejected exactly as an optimizer's candidate would be.

    Set by ``run_ds1000.py --prompts`` to run a config with an optimized
    candidate; nothing else should touch it.
    """
    _OVERRIDES.clear()
    _OVERRIDES.update(mapping or {})


def clear_overrides() -> None:
    _OVERRIDES.clear()


def active_overrides() -> Dict[str, str]:
    return dict(_OVERRIDES)


def _identifiers(text: str) -> FrozenSet[str]:
    """Placeholder names in `text`; raises on a bare, unescaped '$'."""
    names = set()
    for match in string.Template.pattern.finditer(text):
        name = match.group("named") or match.group("braced")
        if name is not None:
            names.add(name)
        elif match.group("invalid") is not None:
            raise ValueError(
                "invalid '$' in prompt text at position "
                f"{match.start()} — write '$$' for a literal dollar sign"
            )
    return frozenset(names)


class Prompt:
    """A named prompt template, mutable in place, with rollback history.

    Mutation is in place by design: components hold a reference to the store, so
    an optimizer's candidate reaches the component without rebuilding anything.
    Returning a new object instead would leave the component on the old one.
    """

    def __init__(self, text: str, name: str = "", optimizable: bool = True):
        self.name = name or "prompt"
        self.optimizable = bool(optimizable)
        self.required = _identifiers(text)
        self.text = text
        self.revision = 0
        self._history: List[str] = []

    # ------------------------------------------------------------- mutation
    def set(self, new_text: str) -> None:
        """Replace the text, keeping the placeholder contract intact."""
        if not self.optimizable:
            raise ValueError(
                f"{self.name}: prompt is frozen (optimizable=False) — it is part "
                "of an output-format contract that downstream parsing depends on"
            )
        missing = self.required - _identifiers(new_text)
        if missing:
            raise ValueError(
                f"{self.name}: new text drops placeholders {sorted(missing)}"
            )
        self._history.append(self.text)
        self.text = new_text
        self.revision += 1

    def rollback(self) -> None:
        """Restore the previous text, if any."""
        if self._history:
            self.text = self._history.pop()
            self.revision += 1

    @contextlib.contextmanager
    def temporarily(self, new_text: str):
        """Apply `new_text` for the duration of the block, then restore.

        The rollout primitive: try a candidate, evaluate, restore — even if the
        evaluation raises.
        """
        self.set(new_text)
        try:
            yield self
        finally:
            self.rollback()

    # ------------------------------------------------------------- rendering
    def render(self, **values) -> str:
        try:
            return string.Template(self.text).substitute(**values)
        except KeyError as exc:
            raise KeyError(
                f"{self.name}: no value supplied for placeholder {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"{self.name}: malformed placeholder — {exc}") from exc

    def fingerprint(self) -> str:
        """Short content hash — computed on access, so it tracks mutations."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]

    def __repr__(self) -> str:
        return (f"Prompt(name={self.name!r}, chars={len(self.text)}, "
                f"rev={self.revision}, optimizable={self.optimizable})")


class PromptDiscoverable:
    """Mixin giving a component a per-instance prompt store.

    Adds no constructor requirements: the store is created on first use, so a
    class only needs to add this to its bases and wrap its literals.

    Registration is lazy, which means a prompt on a branch that never executes
    would never be listed. Rule for migrated components: **register every
    variant unconditionally, branch afterwards** — call ``self.prompt(...)`` for
    both sides of a policy switch, then choose between the returned strings.
    """

    @property
    def _prompts(self) -> Dict[str, Prompt]:
        store = self.__dict__.get("_prompt_store")
        if store is None:
            store = {}
            self.__dict__["_prompt_store"] = store
        return store

    def prompt(self, key: str, default: str, optimizable: bool = True) -> str:
        """Register `default` under `key` on first call; return the live text."""
        entry = self._prompts.get(key)
        if entry is None:
            owner = getattr(self, "name", None) or type(self).__name__
            entry = Prompt(default, name=f"{owner}.{key}", optimizable=optimizable)
            self._prompts[key] = entry
            # Register with the default first, then apply any override through
            # set() so it goes through the same placeholder/frozen validation.
            override = _OVERRIDES.get(f"{owner}.{key}", _OVERRIDES.get(key))
            if override is not None and override != default:
                entry.set(override)
        return entry.text

    def render_prompt(self, key, default, /, **values) -> str:
        """`prompt()` followed by placeholder substitution."""
        self.prompt(key, default)
        return self._prompts[key].render(**values)

    def named_prompts(self) -> Iterator[Tuple[str, Prompt]]:
        yield from self._prompts.items()
