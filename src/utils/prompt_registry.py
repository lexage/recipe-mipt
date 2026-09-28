"""Pipeline-wide prompt discovery.

Walks the component inventory a built pipeline carries (``_components``, attached
by ``PipelineBuilder.build``) and collects every prompt declared through
``PromptDiscoverable`` (see ``src/agent_constructor/prompts.py``).

Used by ``run_ds1000.py`` to record what actually ran, and by any external prompt
optimizer:

    genome = snapshot_prompts(pipeline)      # {key: text}
    apply_candidate(pipeline, {"generator.chat_system": new_text})
    ...evaluate...
    apply_candidate(pipeline, genome)        # restore

Keys are ``"<yaml_component_name>.<prompt_key>"`` — stable across runs, unique,
and derived from the config rather than from object identity.

Two caveats, both inherent rather than accidental:

* Registration is lazy, so a prompt only appears once its code path has run.
  Migrated components register every variant unconditionally (see the mixin's
  docstring); call this after a run, not at build time, for query-time prompts.
* Mutation reaches components that render per call. It does NOT retroactively
  affect a generator whose ``generate()`` already ran during pipeline
  construction — drive such a generator directly instead.
"""

from typing import Dict, List, Tuple

from src.agent_constructor.core import Block
from src.agent_constructor.prompts import Prompt, PromptDiscoverable


def _walk(name: str, obj, seen: set, out: List[Tuple[str, Prompt]], depth: int) -> None:
    if obj is None or id(obj) in seen:
        return
    seen.add(id(obj))

    if isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            _walk(f"{name}[{index}]", item, seen, out, depth)
        return

    if isinstance(obj, PromptDiscoverable):
        for key, prompt in obj.named_prompts():
            out.append((f"{name}.{key}", prompt))

    # One level down into nested blocks (sub-agents a component builds itself).
    if depth > 0:
        for attr, value in vars(obj).items():
            if attr.startswith("_"):
                continue
            if isinstance(value, (Block, PromptDiscoverable)):
                _walk(f"{name}.{attr}", value, seen, out, depth - 1)


def collect_prompts(pipeline, depth: int = 1) -> Dict[str, Prompt]:
    """All prompts declared by the pipeline's components, keyed by path."""
    components = getattr(pipeline, "_components", None) or {}
    out: List[Tuple[str, Prompt]] = []
    seen: set = set()
    for name, component in components.items():
        _walk(name, component, seen, out, depth)
    return dict(out)


def snapshot_prompts(pipeline) -> Dict[str, str]:
    """Current prompt texts — the genome an optimizer reads and restores."""
    return {key: prompt.text for key, prompt in collect_prompts(pipeline).items()}


def apply_candidate(pipeline, mapping: Dict[str, str]) -> None:
    """Apply ``{key: text}`` in place. Unknown keys and no-op writes are errors.

    Fails loudly: a candidate naming a prompt that does not exist means the
    optimizer and the pipeline disagree about the search space.
    """
    prompts = collect_prompts(pipeline)
    unknown = sorted(set(mapping) - set(prompts))
    if unknown:
        raise KeyError(f"no such prompt(s) in this pipeline: {unknown}")
    for key, text in mapping.items():
        if prompts[key].text != text:
            prompts[key].set(text)


def describe_prompts(pipeline) -> Dict[str, Dict]:
    """Compact, JSON-friendly inventory for runtime_stats.json."""
    return {
        key: {
            "name": prompt.name,
            "fingerprint": prompt.fingerprint(),
            "chars": len(prompt.text),
            "optimizable": prompt.optimizable,
            "revision": prompt.revision,
        }
        for key, prompt in collect_prompts(pipeline).items()
    }
