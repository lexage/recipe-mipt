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
            "1. Inspect the repository and identify the code, tests, API definitions, "
            "and usages relevant to the reported behavior. Treat issue text and code "
            "snippets as evidence or reproduction examples; do not assume they are "
            "existing repository code.\n"
            "2. Before editing, establish a baseline: run an existing focused test that "
            "demonstrates the issue or, if no suitable test is available, run a minimal "
            "reproduction using the repository's public API; do not create replacement "
            "test files.\n"
            "3. Read the complete target implementation before changing it. Do not infer "
            "behavior from an incomplete excerpt.\n"
            "4. Apply the smallest source change that addresses the demonstrated cause.\n"
            "5. After the edit, rerun the same baseline command. A missing test path, zero "
            "collected tests, command startup failure, or timeout does not validate the "
            "change. Locate the correct test or use a valid reproduction.\n"
            "6. Run relevant regression tests when available. If validation fails, "
            "inspect the failure and revise the implementation. If dependencies are "
            "unavailable, do not install arbitrary network packages.\n"
            "7. Review the final git diff and remove unrelated changes before finishing. "
            "A valid diff confirms that an edit was made, not that the behavior is correct.\n"
            "Finish only after completing the investigation, implementation, validation, "
            "and final diff review, and only when the patch is non-empty. If validation is "
            "genuinely unavailable, acknowledge that limitation in the final reasoning "
            "instead of claiming that the change was verified.\n\n"
            "Constraints:\n"
            "- Use the editing tools for file changes; do not edit through shell redirection.\n"
            "- The task environment may be offline; do not install arbitrary network dependencies.\n"
            "- Do not create commits or branches.\n"
            "- Do not add or modify tests."
        )
        return "\n\n".join(sections)
