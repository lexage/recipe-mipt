"""Shared contracts and helpers for SWE-rebench critique methods.

The existing DS-1000 critics deliberately do not depend on this module.  The
classes below model a repository patch and its validation evidence instead of
the standalone code response used by DS-1000.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from openai import OpenAI

from src.agent_constructor.agent import Agent


class ReviewDecision(str, Enum):
    """Decision shared by the SWE-rebench tool adapters."""

    ACCEPT = "accept"
    REVISE = "revise"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class MethodologyContract:
    """Non-negotiable stages inherited from the source method."""

    method: str
    stages: tuple[str, ...]
    tool_mode_scope: str
    fidelity_requirements: tuple[str, ...]


METHODOLOGY_CONTRACTS: Mapping[str, MethodologyContract] = {
    "decrim": MethodologyContract(
        method="decrim",
        stages=("decompose", "critique", "refine"),
        tool_mode_scope=(
            "The tool performs decomposition and critique and returns targeted "
            "refinement guidance. The calling agent performs the repository edit."
        ),
        fidelity_requirements=(
            "Decompose the original issue into explicit constraints.",
            "Evaluate every constraint against repository evidence.",
            "Use unmet constraints to drive the next revision.",
        ),
    ),
    "critic": MethodologyContract(
        method="critic",
        stages=("tool_interaction", "critique", "correction"),
        tool_mode_scope=(
            "The tool actively inspects the bound repository and returns an "
            "evidence-grounded critique. The calling agent applies the correction."
        ),
        fidelity_requirements=(
            "Interact with at least one external repository tool.",
            "Ground findings in tool observations.",
            "Use the critique to correct the current candidate patch.",
        ),
    ),
    "self_refine": MethodologyContract(
        method="self_refine",
        stages=("initial_output", "self_feedback", "refinement"),
        tool_mode_scope=(
            "The configured solver model produces self-feedback and concrete "
            "refinement guidance. The calling agent applies that guidance."
        ),
        fidelity_requirements=(
            "Use the same model identity for solving, feedback, and refinement.",
            "Condition refinement on the model's own feedback.",
            "Revalidate the repository after applying the refinement.",
        ),
    ),
    "reflexion": MethodologyContract(
        method="reflexion",
        stages=(
            "trial",
            "environment_feedback",
            "verbal_reflection",
            "episodic_memory",
            "next_trial",
        ),
        tool_mode_scope=(
            "The tool turns test and tool feedback into a verbal reflection and "
            "stores it for the next call for the same task."
        ),
        fidelity_requirements=(
            "Use an environment feedback signal, not an unsupported correctness guess.",
            "Persist verbal reflection in task-scoped episodic memory.",
            "Feed the stored reflection into the next trial.",
        ),
    ),
}


@dataclass(frozen=True)
class SWERebenchCritiqueContext:
    """Repository-level candidate and evidence supplied to a critic."""

    task: str
    git_diff: str
    test_results: str
    repository_evidence: str = ""
    validation_limitations: str = ""
    trajectory_summary: str = ""

    def __post_init__(self) -> None:
        values = {
            "task": self.task,
            "git_diff": self.git_diff,
            "test_results": self.test_results,
            "repository_evidence": self.repository_evidence,
            "validation_limitations": self.validation_limitations,
            "trajectory_summary": self.trajectory_summary,
        }
        for name, value in values.items():
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
        if not self.task.strip():
            raise ValueError("task must contain the original SWE-rebench issue")

    @property
    def task_key(self) -> str:
        return hashlib.sha256(self.task.encode("utf-8")).hexdigest()

    def render(self, *, include_trajectory: bool = False) -> str:
        sections = [
            ("ORIGINAL ISSUE", self.task),
            ("CURRENT COMPLETE GIT DIFF", self.git_diff or "[not provided]"),
            (
                "VALIDATION COMMANDS AND RESULTS",
                self.test_results or "[not provided]",
            ),
            (
                "ADDITIONAL REPOSITORY EVIDENCE",
                self.repository_evidence or "[not provided]",
            ),
            (
                "VALIDATION LIMITATIONS",
                self.validation_limitations or "[none reported]",
            ),
        ]
        if include_trajectory:
            sections.append(
                (
                    "TRAJECTORY SUMMARY",
                    self.trajectory_summary or "[not provided]",
                )
            )
        return "\n\n".join(f"## {title}\n{content}" for title, content in sections)


@dataclass
class CritiqueResult:
    """Stable envelope around method-specific critique output."""

    method: str
    decision: ReviewDecision
    payload: Mapping[str, Any]
    next_required_action: str
    completed_stages: Sequence[str] = field(default_factory=tuple)

    def to_json(self) -> str:
        contract = METHODOLOGY_CONTRACTS[self.method]
        return json.dumps(
            {
                "method": self.method,
                "benchmark": "swe_rebench",
                "mode": "tool",
                "decision": self.decision.value,
                "completed_stages": list(self.completed_stages),
                "methodology_contract": {
                    "stages": list(contract.stages),
                    "tool_mode_scope": contract.tool_mode_scope,
                    "fidelity_requirements": list(contract.fidelity_requirements),
                },
                "result": self.payload,
                "next_required_action": self.next_required_action,
            },
            ensure_ascii=False,
            indent=2,
        )


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one model JSON object while preserving malformed output as evidence."""

    if not isinstance(text, str):
        return {"raw_response": str(text), "parse_error": "response is not text"}
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    candidates = [0]
    candidates.extend(index for index, char in enumerate(stripped) if char == "{")
    for start in dict.fromkeys(candidates):
        try:
            value, _ = decoder.raw_decode(stripped[start:])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return {
        "raw_response": text,
        "parse_error": "model response did not contain a JSON object",
    }


def decision_from_payload(payload: Mapping[str, Any]) -> ReviewDecision:
    value = str(payload.get("decision", "")).strip().lower().replace("-", "_")
    aliases = {
        "accept": ReviewDecision.ACCEPT,
        "accepted": ReviewDecision.ACCEPT,
        "pass": ReviewDecision.ACCEPT,
        "passed": ReviewDecision.ACCEPT,
        "revise": ReviewDecision.REVISE,
        "revision_required": ReviewDecision.REVISE,
        "reject": ReviewDecision.REVISE,
        "failed": ReviewDecision.REVISE,
        "insufficient_evidence": ReviewDecision.INSUFFICIENT_EVIDENCE,
        "unknown": ReviewDecision.INSUFFICIENT_EVIDENCE,
        "uncertain": ReviewDecision.INSUFFICIENT_EVIDENCE,
    }
    return aliases.get(value, ReviewDecision.INSUFFICIENT_EVIDENCE)


class TaskEpisodicMemory:
    """Small task-isolated LRU store used by the Reflexion adapter."""

    def __init__(
        self,
        max_tasks: int = 128,
        max_reflections_per_task: int = 5,
        max_reflection_chars: int = 8_000,
    ):
        if min(max_tasks, max_reflections_per_task, max_reflection_chars) < 1:
            raise ValueError("episodic memory limits must be positive")
        self.max_tasks = max_tasks
        self.max_reflections_per_task = max_reflections_per_task
        self.max_reflection_chars = max_reflection_chars
        self._items: OrderedDict[str, list[str]] = OrderedDict()

    def get(self, task_key: str) -> list[str]:
        values = self._items.get(task_key, [])
        if task_key in self._items:
            self._items.move_to_end(task_key)
        return values.copy()

    def append(self, task_key: str, reflection: str) -> list[str]:
        values = self._items.setdefault(task_key, [])
        values.append(reflection[: self.max_reflection_chars])
        del values[: -self.max_reflections_per_task]
        self._items.move_to_end(task_key)
        while len(self._items) > self.max_tasks:
            self._items.popitem(last=False)
        return values.copy()

    def clear(self, task_key: str) -> None:
        self._items.pop(task_key, None)


class SWERebenchCritiqueAgent(Agent):
    """Base OpenAI-compatible client shared only by SWE-rebench adapters."""

    def __init__(
        self,
        name: str,
        model_name: str,
        openai_api_base_url: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 300.0,
        max_retries: int = 2,
    ):
        super().__init__(name)
        if not model_name:
            raise ValueError("model_name is required")
        if not openai_api_base_url:
            raise ValueError("openai_api_base_url is required")
        if max_tokens < 1 or timeout <= 0 or max_retries < 0:
            raise ValueError("LLM request limits are invalid")
        self.model_name = model_name
        self.openai_api_base_url = openai_api_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            base_url=openai_api_base_url,
            api_key="vllm",
            timeout=timeout,
            max_retries=max_retries,
        )

    def _complete(self, messages: Iterable[Mapping[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (response.choices[0].message.content or "").strip()

    def _prompt(self, system: str, user: str) -> str:
        return self._complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    @staticmethod
    def _result(
        method: str,
        payload: Mapping[str, Any],
        *,
        next_required_action: str,
        completed_stages: Sequence[str],
    ) -> CritiqueResult:
        return CritiqueResult(
            method=method,
            decision=decision_from_payload(payload),
            payload=payload,
            next_required_action=next_required_action,
            completed_stages=completed_stages,
        )
