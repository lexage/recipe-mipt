"""Self-Refine feedback adapter for SWE-rebench repository patches."""

from __future__ import annotations

import json

from .common import (
    SWERebenchCritiqueAgent,
    SWERebenchCritiqueContext,
    parse_json_object,
)


SELF_FEEDBACK_SYSTEM_PROMPT = """You are the SELF-FEEDBACK stage of Self-Refine
for a software-engineering repository task. Evaluate your current candidate
patch against the original issue, repository evidence, and actual validation
results. Do not rely on hidden tests or invent repository behavior.

Return one JSON object and no Markdown:
{
  "decision": "accept|revise|insufficient_evidence",
  "feedback": [
    {
      "priority": "blocking|major|minor",
      "problem": "specific weakness in the current candidate",
      "evidence": "diff, source, or validation evidence",
      "improvement": "how the candidate should be improved"
    }
  ],
  "missing_evidence": ["evidence still needed"]
}
"""


REFINEMENT_SYSTEM_PROMPT = """You are the REFINEMENT stage of Self-Refine.
Use your own self-feedback to produce a concrete, minimal repository refinement
plan. You are running inside a critique tool and cannot edit files directly, so
the calling agent will apply your plan with repository editing tools.

Return one JSON object and no Markdown:
{
  "decision": "accept|revise|insufficient_evidence",
  "refinement_plan": [
    {
      "order": 1,
      "target": "repository path or behavior",
      "change": "precise correction",
      "reason": "feedback item addressed"
    }
  ],
  "validation_plan": ["focused commands or checks after editing"],
  "preserve": ["existing behaviors that must not regress"]
}

If self-feedback found no supported defect and validation is sufficient, return
decision=accept with an empty refinement_plan.
"""


class SWERebenchSelfRefine(SWERebenchCritiqueAgent):
    """Generate self-feedback and refinement guidance with one model identity."""

    def __init__(
        self,
        *args,
        solver_model_name: str | None = None,
        solver_url: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if solver_model_name is not None and solver_model_name != self.model_name:
            raise ValueError(
                "Self-Refine requires solver_model_name to match model_name"
            )
        if solver_url is not None:
            normalized_solver = solver_url.rstrip("/")
            normalized_feedback = self.openai_api_base_url.rstrip("/")
            if normalized_solver != normalized_feedback:
                raise ValueError("Self-Refine requires solver_url to match critic url")
        self.solver_model_name = solver_model_name or self.model_name
        self.solver_url = solver_url or self.openai_api_base_url

    def run(self, context: SWERebenchCritiqueContext):
        rendered_context = context.render()
        feedback_response = self._prompt(
            SELF_FEEDBACK_SYSTEM_PROMPT,
            rendered_context,
        )
        feedback = parse_json_object(feedback_response)

        refinement_response = self._prompt(
            REFINEMENT_SYSTEM_PROMPT,
            (
                f"Candidate and evidence:\n{rendered_context}\n\n"
                "Self-feedback from the same model:\n"
                f"{json.dumps(feedback, ensure_ascii=False, indent=2)}"
            ),
        )
        refinement = parse_json_object(refinement_response)
        decision = refinement.get("decision", feedback.get("decision"))
        payload = {
            "decision": decision or "insufficient_evidence",
            "model_identity": {
                "solver_model": self.solver_model_name,
                "feedback_model": self.model_name,
                "refinement_model": self.model_name,
                "same_model": self.solver_model_name == self.model_name,
            },
            "self_feedback": feedback,
            "refinement_guidance": refinement,
        }
        return self._result(
            "self_refine",
            payload,
            next_required_action=(
                "Apply the same model's refinement guidance to the repository, "
                "run the proposed validation, and invoke Self-Refine again on the "
                "updated candidate."
            ),
            completed_stages=("initial_output", "self_feedback", "refinement"),
        )
