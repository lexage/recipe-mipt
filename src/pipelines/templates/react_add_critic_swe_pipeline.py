"""Mandatory critique pipeline for repository-level SWE-rebench tasks."""

from __future__ import annotations

import json
import logging

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.agent_constructor.pipeline import Pipeline


SWE_REVISION_PROMPT = """Continue working on the SWE-rebench task below.

The current repository already contains an initial candidate patch. A mandatory
critic has reviewed that exact patch. Inspect the current git diff, verify the
critic feedback against the task and repository, and apply only supported
corrections. Preserve correct existing changes. If the critic reports missing
evidence, obtain that evidence with repository tools before deciding whether an
edit is needed. Run focused validation after the latest edit, inspect the final
diff, and then finish.

This pipeline performs one critic pass. Do not look for or invoke a critic tool;
apply the supplied feedback directly with the normal repository tools.

ORIGINAL TASK:
{task}

CRITIC RESULT:
{critique}
"""


class REACTPipelineAddCriticSWE(Pipeline):
    """Run initial ReAct, mandatory critique, then one repository revision."""

    _MAX_CONTEXT_CHARS = 50_000

    def __init__(
        self,
        react_agent: Agent,
        critic_agent: Agent,
        solver_agent: Agent,
    ):
        super().__init__("react_add_critic_swe_pipeline")
        self.react_agent = react_agent
        self.critic_agent = critic_agent
        self.solver_agent = solver_agent

    @classmethod
    def _bounded_tail(cls, text: str) -> str:
        if len(text) <= cls._MAX_CONTEXT_CHARS:
            return text
        return (
            "[Earlier observations omitted because the context was too long.]\n"
            + text[-cls._MAX_CONTEXT_CHARS :]
        )

    @classmethod
    def _run_command_observations(cls, agent: Agent) -> str:
        observations = []
        for message in getattr(agent, "memory", []):
            content = message.get("content", "") if isinstance(message, dict) else ""
            if content.startswith("Observation from run_command"):
                observations.append(content)
        if not observations:
            return "[No run_command observations were recorded by the initial agent.]"
        return cls._bounded_tail("\n\n".join(observations))

    @classmethod
    def _validation_limitations(cls, agent: Agent) -> str:
        limitations = []
        if not any(
            isinstance(message, dict)
            and str(message.get("content", "")).startswith(
                "Observation from run_command"
            )
            for message in getattr(agent, "memory", [])
        ):
            limitations.append("The initial agent did not record a run_command result.")

        error_history = getattr(agent, "error_history", [])
        if error_history:
            limitations.append(
                "Initial-agent errors:\n"
                + json.dumps(error_history, ensure_ascii=False, indent=2)
            )
        return cls._bounded_tail("\n\n".join(limitations))

    def _build_context(self, task: str, initial_answer: str):
        from src.agents.critique.swe_rebench import SWERebenchCritiqueContext
        from src.benchmarks.swe_rebench.runtime import get_repository_runtime

        patch = get_repository_runtime().get_patch()
        return SWERebenchCritiqueContext(
            task=task,
            git_diff=patch,
            test_results=self._run_command_observations(self.react_agent),
            repository_evidence=initial_answer,
            validation_limitations=self._validation_limitations(self.react_agent),
            trajectory_summary=str(getattr(self.react_agent, "_rec_mem", "")),
        )

    def run(self, task: Text):
        from src.agents.critique.swe_rebench import CritiqueResult, ReviewDecision

        initial_answer = str(self.react_agent.run(task))
        context = self._build_context(str(task), initial_answer)
        critique = self.critic_agent.run(context)
        if not isinstance(critique, CritiqueResult):
            raise TypeError("critic_agent must return CritiqueResult")

        logging.info(
            "SWE_CRITIC_PIPELINE_DECISION\tmethod=%s\tdecision=%s",
            critique.method,
            critique.decision.value,
        )
        if critique.decision == ReviewDecision.ACCEPT:
            return initial_answer

        revision_task = SWE_REVISION_PROMPT.format(
            task=task,
            critique=critique.to_json(mode="pipeline"),
        )
        return self.solver_agent.run(revision_task)
