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
            "Instructions:\n"
            "- Inspect the repository using the available tools.\n"
            "- Apply the smallest correct fix to the working tree; do not only describe it.\n"
            "- Edit files only with the editing tools, never with shell redirection or sed -i.\n"
            "- Pass raw unified diffs without Markdown fences to apply_patch.\n"
            "- If an edit fails, read the target file again and change the strategy; "
            "do not retry by changing only hunk line numbers.\n"
            "- Run relevant tests in the task environment and treat a non-zero exit code as failure.\n"
            "- Review the final git diff before finishing.\n"
            "- Do not modify tests unless the issue explicitly requires it."
        )
        return "\n\n".join(sections)
