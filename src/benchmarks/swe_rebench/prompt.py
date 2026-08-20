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
            "- Locate the relevant behavior by inspecting source code, existing tests, "
            "and call sites before editing. Treat issue text and code snippets as evidence "
            "or reproduction examples; do not assume they are existing repository code.\n"
            "- Read the complete relevant functions and nearby tests. Do not infer behavior "
            "from names or incomplete excerpts alone.\n"
            "- Use an existing focused test when possible. Temporary reproduction code is "
            "allowed for diagnosis, but remove it before finishing and do not turn "
            "production modules into debug scripts.\n"
            "- Make the smallest source-code change supported by repository evidence.\n"
            "- Run focused validation after editing. A command that collected no tests, "
            "failed before relevant code ran, or exercised no relevant behavior is not "
            "successful validation.\n"
            "- Review the final git diff and remove unintended changes before finishing.\n\n"
            "Constraints:\n"
            "- Use the editing tools for file changes; do not edit through shell redirection.\n"
            "- The task environment may be offline; do not install arbitrary network dependencies.\n"
            "- Do not create commits or branches.\n"
            "- Do not add or modify tests unless the issue explicitly requires it. "
            "Temporary reproduction artifacts must not remain in the final diff."
        )
        return "\n\n".join(sections)
