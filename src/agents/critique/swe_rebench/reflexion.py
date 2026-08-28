"""Reflexion adapter with task-scoped episodic memory for SWE-rebench."""

from __future__ import annotations

import json

from .common import (
    SWERebenchCritiqueAgent,
    SWERebenchCritiqueContext,
    TaskEpisodicMemory,
    parse_json_object,
)


FEEDBACK_SIGNAL_SYSTEM_PROMPT = """You are the environment-feedback interpreter
for a Reflexion coding agent. Derive a feedback signal from actual test results,
tool failures, and the current diff. Do not replace missing execution evidence
with a subjective correctness score and do not claim hidden tests pass.

Return one JSON object and no Markdown:
{
  "outcome": "success|failure|uncertain",
  "score": 1,
  "evidence": ["specific observed signal"],
  "failed_checks": ["failed or inconclusive checks"],
  "missing_evidence": ["feedback that the environment did not provide"]
}

score is an optional 1-5 summary signal; evidence and outcome are authoritative.
"""


REFLECTION_SYSTEM_PROMPT = """You implement the VERBAL REFLECTION stage of
Reflexion. Reflect on the current trial using the environment feedback and prior
task-scoped reflections. Identify failed assumptions and produce a different,
actionable strategy for the next trial. Do not merely restate test output.

Return one JSON object and no Markdown:
{
  "decision": "accept|revise|insufficient_evidence",
  "reflection": "concise verbal lesson retained across trials",
  "failed_assumptions": ["assumption contradicted by evidence"],
  "successful_actions": ["actions worth preserving"],
  "next_strategy": ["ordered actions for the next trial"],
  "avoid_repeating": ["unproductive actions or hypotheses"],
  "validation_plan": ["environment checks for the next trial"]
}
"""


class SWERebenchReflexion(SWERebenchCritiqueAgent):
    """Convert environment feedback into persistent verbal reflection."""

    def __init__(
        self,
        *args,
        max_memory_tasks: int = 128,
        max_reflections_per_task: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.episodic_memory = TaskEpisodicMemory(
            max_tasks=max_memory_tasks,
            max_reflections_per_task=max_reflections_per_task,
        )

    def run(self, context: SWERebenchCritiqueContext):
        previous_reflections = self.episodic_memory.get(context.task_key)
        feedback_response = self._prompt(
            FEEDBACK_SIGNAL_SYSTEM_PROMPT,
            context.render(include_trajectory=True),
        )
        environment_feedback = parse_json_object(feedback_response)

        reflection_response = self._prompt(
            REFLECTION_SYSTEM_PROMPT,
            (
                f"Current trial:\n{context.render(include_trajectory=True)}\n\n"
                "Environment feedback:\n"
                f"{json.dumps(environment_feedback, ensure_ascii=False, indent=2)}\n\n"
                "Prior episodic reflections for this task:\n"
                f"{json.dumps(previous_reflections, ensure_ascii=False, indent=2)}"
            ),
        )
        reflection = parse_json_object(reflection_response)
        memory_text = reflection.get("reflection")
        if not isinstance(memory_text, str) or not memory_text.strip():
            memory_text = json.dumps(reflection, ensure_ascii=False)
        updated_memory = self.episodic_memory.append(
            context.task_key,
            memory_text,
        )

        payload = {
            "decision": reflection.get("decision", "insufficient_evidence"),
            "environment_feedback": environment_feedback,
            "reflection": reflection,
            "episodic_memory": updated_memory,
            "memory_scope": "current_task_only",
        }
        return self._result(
            "reflexion",
            payload,
            next_required_action=(
                "Use the stored reflection and next_strategy in the next repository "
                "trial, obtain a new environment feedback signal, and invoke "
                "Reflexion again."
            ),
            completed_stages=(
                "trial",
                "environment_feedback",
                "verbal_reflection",
                "episodic_memory",
            ),
        )

    def clear_task_memory(self, task: str) -> None:
        context = SWERebenchCritiqueContext(task=task, git_diff="", test_results="")
        self.episodic_memory.clear(context.task_key)
