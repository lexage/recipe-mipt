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
            "- Treat code snippets in the issue as reproduction examples unless repository "
            "evidence shows that they are existing project code. Do not copy an absent example "
            "into the repository as a fix.\n"
            "- Trace the incorrect externally visible behavior to the implementation that "
            "produces it before editing.\n"
            "- Apply the smallest correct fix to the working tree; do not only describe it.\n"
            "- Edit files only with the editing tools, never with shell redirection or sed -i.\n"
            "- Pass raw unified diffs without Markdown fences to apply_patch.\n"
            "- If apply_patch fails, read the target file again, correct the unified diff, "
            "and retry with changed context or strategy; do not change only hunk line numbers.\n"
            "- The task environment is prebuilt and may be offline. Do not install arbitrary "
            "dependencies from the network.\n"
            "- Start with focused tests related to the issue. Missing optional dependencies "
            "in unrelated tests do not by themselves mean that the environment is broken.\n"
            "- Treat a non-zero test exit code as failure and inspect its output.\n"
            "- After editing, perform the most relevant available validation. If a relevant "
            "test cannot run, use the closest meaningful static, import, or focused check and "
            "inspect its result.\n"
            "- Before finishing, call git_diff and verify that the intended patch is non-empty, "
            "all referenced symbols are defined or imported, and the changed code participates "
            "in the execution path responsible for the issue.\n"
            "- A non-empty diff proves only that files changed; it does not prove that the issue "
            "is fixed.\n"
            "- Do not finish after a failed edit or while git_diff shows no changes.\n"
            "- Do not modify tests unless the issue explicitly requires it."
        )
        return "\n\n".join(sections)
