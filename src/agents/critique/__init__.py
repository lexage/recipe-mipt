"""Critique agents with compatibility-preserving lazy imports.

The DS-1000 implementations retain their existing public names. Lazy loading
keeps optional dependencies of one critic from blocking unrelated critics.
"""

from importlib import import_module


_EXPORTS = {
    "ComplexCritic": ("src.agents.critique.complex_critic.main", "ComplexCritic"),
    "Critic": ("src.agents.critique.critic.main", "Critic"),
    "Decrim": ("src.agents.critique.decrim.main", "Decrim"),
    "Reflexion": ("src.agents.critique.reflexion.main", "Reflexion"),
    "SelfRefine": ("src.agents.critique.self_refine.main", "SelfRefine"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
