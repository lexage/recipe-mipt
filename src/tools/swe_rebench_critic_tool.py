"""Tool adapter for SWE-rebench-specific critique methods."""

from __future__ import annotations

import hashlib
import logging

from typing import Any, Dict, Literal

from src.agents.critique.swe_rebench import (
    SWERebenchCritic,
    SWERebenchCritiqueContext,
    SWERebenchDecrim,
    SWERebenchReflexion,
    SWERebenchSelfRefine,
)

from .base_tool import BaseTool, ToolResult


SWE_CRITIC_DESCRIPTIONS = {
    "decrim": (
        "Apply the DeCRIM critique component to the current SWE-rebench patch. "
        "You MUST invoke it after obtaining the complete git diff and relevant "
        "validation and before finishing. "
        "It decomposes the original issue into constraints, evaluates each one, "
        "and returns targeted refinement guidance. Apply that guidance, rerun "
        "validation, and invoke the tool again before finishing."
    ),
    "critic": (
        "Apply tool-interactive CRITIC to the current SWE-rebench patch. You MUST "
        "invoke it after code changes and initial validation and before finishing. "
        "The critic independently uses "
        "the bound repository's inspection and validation tools, then returns an "
        "evidence-grounded review. Correct supported findings and revalidate."
    ),
    "self_refine": (
        "Apply the Self-Refine feedback component to the current SWE-rebench "
        "candidate. You MUST invoke it after producing and validating an initial "
        "patch and before finishing. The same configured solver model generates self-feedback and "
        "refinement guidance. Apply the guidance with repository editing tools, "
        "run its validation plan, and invoke the tool again on the new candidate."
    ),
    "reflexion": (
        "Apply Reflexion to the current SWE-rebench trial. Provide actual test and "
        "tool feedback plus the recent trajectory. You MUST invoke it after a failed "
        "or inconclusive trial before starting the next trial. The tool creates a verbal "
        "reflection and stores it in task-scoped episodic memory for the next trial. "
        "Use next_strategy before invoking Reflexion again."
    ),
}


class SWERebenchCriticTool(BaseTool):
    """Expose a method-faithful SWE-rebench critique component to ReAct."""

    _AGENT_REGISTRY = {
        "decrim": SWERebenchDecrim,
        "critic": SWERebenchCritic,
        "self_refine": SWERebenchSelfRefine,
        "reflexion": SWERebenchReflexion,
    }

    def __init__(
        self,
        name: Literal["decrim", "critic", "self_refine", "reflexion"] = "critic",
        description: str | None = None,
        url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 300.0,
        max_retries: int = 2,
        max_tool_iterations: int = 6,
        max_tool_result_chars: int = 20_000,
        solver_model_name: str | None = None,
        solver_url: str | None = None,
        max_memory_tasks: int = 128,
        max_reflections_per_task: int = 5,
    ):
        if name not in self._AGENT_REGISTRY:
            raise ValueError(
                f"Unknown SWE-rebench critique method: '{name}'. Available: "
                f"{', '.join(self._AGENT_REGISTRY)}"
            )
        super().__init__(
            name=name,
            description=description or SWE_CRITIC_DESCRIPTIONS[name],
        )

        common_kwargs = {
            "name": f"swe-rebench-{name}",
            "model_name": model_name,
            "openai_api_base_url": url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if name == "critic":
            common_kwargs.update(
                {
                    "max_tool_iterations": max_tool_iterations,
                    "max_tool_result_chars": max_tool_result_chars,
                }
            )
        elif name == "self_refine":
            common_kwargs.update(
                {
                    "solver_model_name": solver_model_name,
                    "solver_url": solver_url,
                }
            )
        elif name == "reflexion":
            common_kwargs.update(
                {
                    "max_memory_tasks": max_memory_tasks,
                    "max_reflections_per_task": max_reflections_per_task,
                }
            )
        self.critic_agent = self._AGENT_REGISTRY[name](**common_kwargs)
        self._last_diff_hash_by_task: dict[str, str] = {}

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": (
                                "The original SWE-rebench issue exactly as given to "
                                "the solving agent."
                            ),
                        },
                        "git_diff": {
                            "type": "string",
                            "description": (
                                "The complete current git diff after the most recent "
                                "repository edit, including every changed file."
                            ),
                        },
                        "test_results": {
                            "type": "string",
                            "description": (
                                "Relevant validation commands and their complete "
                                "results, including exit codes and failures."
                            ),
                        },
                        "repository_evidence": {
                            "type": "string",
                            "description": (
                                "Optional source, test, and call-site evidence used "
                                "to justify the candidate patch."
                            ),
                        },
                        "validation_limitations": {
                            "type": "string",
                            "description": (
                                "Optional environment errors, checks that could not "
                                "run, and unsupported assumptions."
                            ),
                        },
                        "trajectory_summary": {
                            "type": "string",
                            "description": (
                                "Optional recent action/observation summary. Required "
                                "for useful Reflexion after a failed trial."
                            ),
                        },
                    },
                    "required": ["task", "git_diff", "test_results"],
                },
            },
        }

    def __call__(
        self,
        task: str,
        git_diff: str,
        test_results: str,
        repository_evidence: str = "",
        validation_limitations: str = "",
        trajectory_summary: str = "",
    ) -> ToolResult:
        try:
            context = SWERebenchCritiqueContext(
                task=task,
                git_diff=git_diff,
                test_results=test_results,
                repository_evidence=repository_evidence,
                validation_limitations=validation_limitations,
                trajectory_summary=trajectory_summary,
            )
            diff_hash = hashlib.sha256(git_diff.encode("utf-8")).hexdigest()
            previous_diff_hash = self._last_diff_hash_by_task.get(context.task_key)
            logging.info(
                "SWE_CRITIQUE_START\tmethod=%s\ttask_key=%s\t"
                "candidate_changed_since_previous_critique=%s",
                self.name,
                context.task_key[:12],
                previous_diff_hash is not None and previous_diff_hash != diff_hash,
            )
            result = self.critic_agent.run(context)
            self._last_diff_hash_by_task[context.task_key] = diff_hash
            logging.info(
                "SWE_CRITIQUE_DONE\tmethod=%s\ttask_key=%s\tdecision=%s",
                self.name,
                context.task_key[:12],
                result.decision.value,
            )
            return ToolResult.ok(result.to_json(), progress=True)
        except Exception as error:
            logging.exception("SWE_CRITIQUE_FAILED\tmethod=%s", self.name)
            return ToolResult.error(
                f"SWE-rebench {self.name} critique failed: "
                f"{type(error).__name__}: {error}",
                error_code="swe_rebench_critique_failed",
            )
