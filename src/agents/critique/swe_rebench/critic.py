"""Tool-interactive CRITIC adapter for SWE-rebench repositories."""

from __future__ import annotations

from typing import Any, Mapping

from src.tools.base_tool import BaseTool, ToolResult
from src.tools.git_diff import GitDiffTool
from src.tools.list_files import ListFilesTool
from src.tools.read_file import ReadFileTool
from src.tools.run_command import RunCommandTool
from src.tools.search_code import SearchCodeTool

from .common import (
    SWERebenchCritiqueAgent,
    SWERebenchCritiqueContext,
    parse_json_object,
)


CRITIC_SYSTEM_PROMPT = """You implement CRITIC: tool-interactive critiquing for
a SWE-rebench repository patch. You must actively use at least one available
repository tool before producing a final critique. Prefer git_diff first, then
inspect the changed code, relevant call sites, and focused validation evidence.
You may inspect and test but you must never edit the repository.

At every turn return exactly one JSON object and no Markdown:
{
  "thought": "what evidence is needed next",
  "action": "one exact tool name or finish",
  "action_input": {"argument": "value"},
  "decision": "accept|revise|insufficient_evidence",
  "critique": [
    {
      "severity": "blocking|major|minor",
      "finding": "specific problem",
      "evidence": "tool-grounded evidence",
      "required_change": "actionable correction"
    }
  ],
  "validation_required": ["checks still required"]
}

For a tool action, critique may be empty while evidence is collected. For
action=finish, action_input must be empty and every finding must cite an actual
tool observation. Do not claim hidden tests pass.

Available tools:
{tools}
"""


class SWERebenchCritic(SWERebenchCritiqueAgent):
    """Run an evidence-gathering CRITIC loop without repository edit tools."""

    def __init__(
        self,
        *args,
        max_tool_iterations: int = 6,
        max_tool_result_chars: int = 20_000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if max_tool_iterations < 1 or max_tool_result_chars < 1:
            raise ValueError("CRITIC tool limits must be positive")
        self.max_tool_iterations = max_tool_iterations
        self.max_tool_result_chars = max_tool_result_chars
        repository_tools: list[BaseTool] = [
            ListFilesTool(),
            ReadFileTool(),
            SearchCodeTool(),
            RunCommandTool(allow_package_install=False),
            GitDiffTool(),
        ]
        self.repository_tools = {tool.name: tool for tool in repository_tools}

    def _tools_prompt(self) -> str:
        return "\n".join(
            tool.get_prompt_description() for tool in self.repository_tools.values()
        )

    @staticmethod
    def _tool_succeeded(result: Any) -> bool:
        return not isinstance(result, ToolResult) or result.success

    def _execute_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> tuple[str, bool]:
        tool = self.repository_tools.get(name)
        if tool is None:
            return (
                f"Unknown tool '{name}'. Available: {', '.join(self.repository_tools)}",
                False,
            )
        if not isinstance(arguments, dict):
            return "action_input must be a JSON object", False
        validation_error = tool.validate_arguments(arguments)
        if validation_error:
            return f"Tool argument validation failed: {validation_error}", False
        try:
            result = tool(**arguments)
        except Exception as error:
            return f"Tool execution failed: {type(error).__name__}: {error}", False
        rendered = str(result)
        if len(rendered) > self.max_tool_result_chars:
            rendered = (
                rendered[: self.max_tool_result_chars]
                + "\n[tool observation truncated]"
            )
        return rendered, self._tool_succeeded(result)

    def run(self, context: SWERebenchCritiqueContext):
        system = CRITIC_SYSTEM_PROMPT.replace("{tools}", self._tools_prompt())
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Review this candidate. Supplied evidence is context, but you "
                    "must independently interact with repository tools before finish.\n\n"
                    + context.render()
                ),
            },
        ]
        tool_interactions: list[dict[str, Any]] = []
        successful_tool_calls = 0

        for _ in range(self.max_tool_iterations):
            response = self._complete(messages)
            step = parse_json_object(response)
            action = str(step.get("action", "")).strip()
            action_input = step.get("action_input", {})
            messages.append({"role": "assistant", "content": response})

            if action.lower() == "finish":
                if not tool_interactions:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "CRITIC requires external tool interaction. Use at "
                                "least one repository tool before finishing."
                            ),
                        }
                    )
                    continue
                payload = {
                    **step,
                    "tool_interactions": tool_interactions,
                    "successful_tool_calls": successful_tool_calls,
                }
                return self._result(
                    "critic",
                    payload,
                    next_required_action=(
                        "Correct every supported finding, rerun the relevant "
                        "validation, and invoke CRITIC again on the updated patch."
                    ),
                    completed_stages=("tool_interaction", "critique"),
                )

            observation, success = self._execute_tool(action, action_input)
            if action in self.repository_tools:
                tool_interactions.append(
                    {
                        "tool": action,
                        "arguments": action_input,
                        "success": success,
                        "observation_excerpt": observation[:4_000],
                    }
                )
                successful_tool_calls += int(success)
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation from {action}:\n{observation}",
                }
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    "The repository-tool budget is exhausted. Return action=finish "
                    "with a final evidence-grounded critique now."
                ),
            }
        )
        final_step = parse_json_object(self._complete(messages))
        final_step["action"] = "finish"
        final_step["tool_interactions"] = tool_interactions
        final_step["successful_tool_calls"] = successful_tool_calls
        if not tool_interactions:
            final_step["decision"] = "insufficient_evidence"
            final_step["methodology_error"] = (
                "CRITIC could not complete its required external tool interaction"
            )
        return self._result(
            "critic",
            final_step,
            next_required_action=(
                "Use the grounded findings to revise the patch and rerun focused "
                "validation before finishing."
            ),
            completed_stages=("tool_interaction", "critique")
            if tool_interactions
            else ("critique",),
        )
