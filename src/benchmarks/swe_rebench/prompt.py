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
            "- Treat issue text and code snippets as evidence or reproduction examples; "
            "do not assume they are existing repository code.\n"
            "- Before editing, inspect the complete target implementation and existing "
            "relevant tests or usages.\n"
            "- Apply the smallest source change that addresses the demonstrated cause.\n"
            "- Run existing focused tests when available. If they cannot run because "
            "dependencies are unavailable, do not install packages and do not create "
            "replacement test files.\n"
            "- Review the final git diff and remove unrelated changes before finishing.\n"
            "- Finish only when the patch is non-empty.\n\n"
            "Constraints:\n"
            "- Use the editing tools for file changes; do not edit through shell redirection.\n"
            "- The task environment may be offline; do not install arbitrary network dependencies.\n"
            "- Do not create commits or branches.\n"
            "- Do not add or modify tests."
        )
        return "\n\n".join(sections)
