"""Pipeline-facing wrapper around the SWE-rebench critique methods."""

from __future__ import annotations

import logging

from typing import Literal

from src.agent_constructor.agent import Agent

from .common import CritiqueResult, SWERebenchCritiqueContext
from .critic import SWERebenchCritic
from .decrim import SWERebenchDecrim
from .reflexion import SWERebenchReflexion
from .self_refine import SWERebenchSelfRefine


class SWERebenchPipelineCritic(Agent):
    """Select and run one existing SWE-rebench critique implementation."""

    _AGENT_REGISTRY = {
        "decrim": SWERebenchDecrim,
        "critic": SWERebenchCritic,
        "self_refine": SWERebenchSelfRefine,
        "reflexion": SWERebenchReflexion,
    }

    def __init__(
        self,
        method: Literal["decrim", "critic", "self_refine", "reflexion"],
        url: str,
        model_name: str,
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
        if method not in self._AGENT_REGISTRY:
            raise ValueError(
                f"Unknown SWE-rebench critique method: '{method}'. Available: "
                f"{', '.join(self._AGENT_REGISTRY)}"
            )
        super().__init__(f"swe-rebench-pipeline-{method}")
        self.method = method

        critic_kwargs = {
            "name": self.name,
            "model_name": model_name,
            "openai_api_base_url": url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if method == "critic":
            critic_kwargs.update(
                {
                    "max_tool_iterations": max_tool_iterations,
                    "max_tool_result_chars": max_tool_result_chars,
                }
            )
        elif method == "self_refine":
            critic_kwargs.update(
                {
                    "solver_model_name": solver_model_name,
                    "solver_url": solver_url,
                }
            )
        elif method == "reflexion":
            critic_kwargs.update(
                {
                    "max_memory_tasks": max_memory_tasks,
                    "max_reflections_per_task": max_reflections_per_task,
                }
            )

        self.critic = self._AGENT_REGISTRY[method](**critic_kwargs)

    def run(self, context: SWERebenchCritiqueContext) -> CritiqueResult:
        if not isinstance(context, SWERebenchCritiqueContext):
            raise TypeError("SWERebenchPipelineCritic expects a critique context")
        logging.info(
            "SWE_PIPELINE_CRITIQUE_START\tmethod=%s\ttask_key=%s",
            self.method,
            context.task_key[:12],
        )
        result = self.critic.run(context)
        logging.info(
            "SWE_PIPELINE_CRITIQUE_DONE\tmethod=%s\tdecision=%s",
            self.method,
            result.decision.value,
        )
        return result
