"""DeCRIM critique adapter for repository-editing SWE-rebench tasks."""

from __future__ import annotations

import json

from .common import (
    SWERebenchCritiqueAgent,
    SWERebenchCritiqueContext,
    parse_json_object,
)


DECOMPOSE_SYSTEM_PROMPT = """You implement the DECOMPOSE stage of DeCRIM.
Decompose only the original software-engineering issue into independently
checkable constraints. Preserve explicit requirements and cautiously identify
implicit compatibility or scope constraints; do not invent requirements from a
candidate patch. Return one JSON object and no Markdown.

Schema:
{
  "constraints": [
    {
      "id": "C1",
      "requirement": "checkable requirement",
      "kind": "functional|compatibility|scope|validation",
      "source": "explicit|implicit",
      "rationale": "why this follows from the issue"
    }
  ]
}
"""


CRITIQUE_SYSTEM_PROMPT = """You implement the CRITIQUE stage of DeCRIM for a
SWE-rebench repository patch. Evaluate every supplied constraint independently
against the complete diff, validation results, and repository evidence. Do not
claim hidden tests pass and do not treat a command that collected no tests or
failed before exercising the behavior as successful validation.

Return one JSON object and no Markdown using this schema:
{
  "decision": "accept|revise|insufficient_evidence",
  "constraint_reviews": [
    {
      "id": "C1",
      "status": "satisfied|not_satisfied|insufficient_evidence",
      "evidence": ["specific diff, source, or validation evidence"],
      "reason": "short explanation",
      "required_change": "targeted change or empty string"
    }
  ],
  "refinement_guidance": ["ordered actions for unmet constraints"],
  "validation_required": ["checks to run after refinement"]
}

Set decision=accept only when every constraint is supported by evidence.
"""


class SWERebenchDecrim(SWERebenchCritiqueAgent):
    """Decompose an issue, critique its patch, and return refinement guidance."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._constraints_by_task: dict[str, dict] = {}

    def _decompose(self, context: SWERebenchCritiqueContext) -> dict:
        cached = self._constraints_by_task.get(context.task_key)
        if cached is not None:
            return cached
        response = self._prompt(
            DECOMPOSE_SYSTEM_PROMPT,
            f"Original issue:\n{context.task}",
        )
        constraints = parse_json_object(response)
        raw_constraints = constraints.get("constraints")
        if not isinstance(raw_constraints, list) or not raw_constraints:
            constraints = {
                "constraints": [],
                "decomposition_error": constraints,
            }
        if constraints.get("constraints"):
            self._constraints_by_task[context.task_key] = constraints
        return constraints

    def run(self, context: SWERebenchCritiqueContext):
        constraints = self._decompose(context)
        critique_response = self._prompt(
            CRITIQUE_SYSTEM_PROMPT,
            (
                f"Decomposed constraints:\n{json.dumps(constraints, ensure_ascii=False, indent=2)}"
                f"\n\nCandidate and evidence:\n{context.render()}"
            ),
        )
        critique = parse_json_object(critique_response)
        payload = {
            "decomposition": constraints,
            "critique": critique,
        }
        decision_payload = {
            "decision": critique.get("decision", "insufficient_evidence")
        }
        result = self._result(
            "decrim",
            {**payload, **decision_payload},
            next_required_action=(
                "Apply the refinement guidance for every unmet constraint, run "
                "focused validation, collect a new complete git diff, and invoke "
                "DeCRIM again before finishing."
            ),
            completed_stages=("decompose", "critique"),
        )
        return result
