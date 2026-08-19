"""Prompt construction for repository-editing SWE-rebench agents."""

from dataclasses import dataclass

from .data_types import SWERebenchTask


@dataclass(frozen=True)
class SWERebenchPromptBuilder:
    include_hints: bool = False

    def build(self, task: SWERebenchTask, workdir: str = "/testbed") -> str:
        if not isinstance(task, SWERebenchTask):
            raise TypeError("task must be a SWERebenchTask")
        if not isinstance(workdir, str) or not workdir.startswith("/"):
            raise ValueError("workdir must be an absolute POSIX path")

        sections = [
            "You are solving a software engineering task in an isolated repository.",
            f"Instance ID: {task.instance_id}",
            f"Repository: {task.repo}",
            f"Base commit: {task.base_commit}",
            f"Repository workdir: {workdir}",
            f"Issue:\n{task.problem_statement}",
        ]
        if self.include_hints and task.hints_text:
            sections.append(f"Hints:\n{task.hints_text}")
        sections.append(
            "Goal:\n"
            "Produce the smallest repository patch that fixes the reported behavior.\n\n"
            "Workflow:\n"
            "- Understand the expected and current behavior before editing.\n"
            "- Treat issue text and code snippets as evidence or reproduction examples; "
            "do not assume they are existing repository code.\n"
            "- Locate the responsible implementation and inspect nearby tests or usages.\n"
            "- Form a concrete causal hypothesis supported by repository evidence.\n"
            "- Apply the smallest change that addresses that cause.\n"
            "- If an operation repeatedly fails, stop repeating it and change strategy.\n"
            "- Validate the changed behavior with the most relevant available test or check.\n"
            "- Review the final git diff for correctness, scope, imports, and unintended changes.\n"
            "- Finish only when the patch is non-empty. Attempt relevant validation when "
            "available; if it cannot run, inspect the failure and avoid unrelated changes.\n\n"
            "Constraints:\n"
            "- Use the editing tools for file changes; do not edit through shell redirection.\n"
            "- The task environment may be offline; do not install arbitrary network dependencies.\n"
            "- Do not create commits or branches.\n"
            "- Do not modify tests unless the issue explicitly requires it."
        )
        return "\n\n".join(sections)
