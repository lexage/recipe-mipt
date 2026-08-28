# SWE-rebench critique adapters

This package adapts the existing critique methods to repository-level
SWE-rebench candidates without changing the DS-1000 implementations in the
parent directories.

The adapters are exposed as tools. In tool mode the calling ReAct agent owns the
repository edit and the subsequent validation. Consequently, a tool call alone
is only the critique/reflection part of a method: methodological completion also
requires the caller to apply the returned guidance and run a new trial.

Method contracts are declared in `common.py`:

- DeCRIM decomposes the original issue, evaluates every constraint, and returns
  targeted refinement guidance.
- CRITIC actively uses read-only or validation repository tools before producing
  an evidence-grounded critique.
- Self-Refine uses the configured solver model for self-feedback and refinement
  guidance; its configuration rejects a different solver model or endpoint.
- Reflexion derives a feedback signal from the environment, writes a verbal
  reflection, and persists it in task-scoped episodic memory for the next trial.

The separate mandatory pipeline orchestration is intentionally not implemented
in this package.
