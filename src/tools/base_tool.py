from abc import abstractmethod
from typing import Any, Dict, Optional

from src.agent_constructor.core import Block


class ToolResult(str):
    """String-compatible result carrying generic tool execution status."""

    success: bool
    error_code: Optional[str]
    retryable: bool
    progress: Optional[bool]
    validation_status: Optional[str]

    def __new__(
        cls,
        content: str,
        *,
        success: bool = True,
        error_code: Optional[str] = None,
        retryable: bool = True,
        progress: Optional[bool] = None,
        validation_status: Optional[str] = None,
    ) -> "ToolResult":
        instance = super().__new__(cls, content)
        instance.success = success
        instance.error_code = error_code
        instance.retryable = retryable
        instance.progress = progress
        instance.validation_status = validation_status
        return instance

    @classmethod
    def ok(
        cls,
        content: str,
        *,
        progress: Optional[bool] = None,
        validation_status: Optional[str] = None,
    ) -> "ToolResult":
        """Return a successful result with optional semantic-progress metadata."""

        return cls(
            content,
            success=True,
            progress=progress,
            validation_status=validation_status,
        )

    @classmethod
    def error(
        cls,
        content: str,
        *,
        error_code: str = "tool_error",
        retryable: bool = True,
        validation_status: Optional[str] = None,
    ) -> "ToolResult":
        return cls(
            content,
            success=False,
            error_code=error_code,
            retryable=retryable,
            progress=False,
            validation_status=validation_status,
        )


class BaseTool(Block):
    """Абстрактный класс инструмента."""

    def __init__(
        self,
        name: str = "",
        description: str = "",
        *,
        mutates_repository: bool = False,
        provides_diff: bool = False,
        repository_path_args: tuple[str, ...] = (),
    ):
        self.name = name
        self.description = description
        self.mutates_repository = mutates_repository
        self.provides_diff = provides_diff
        self.repository_path_args = repository_path_args

    def validate_arguments(self, arguments: Dict[str, Any]) -> Optional[str]:
        """Validate the JSON-schema subset used by repository tools."""

        if not isinstance(arguments, dict):
            return "action_input must be an object"
        parameters = self.get_schema().get("function", {}).get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        missing = [name for name in required if name not in arguments]
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}"
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            return f"Unexpected argument(s): {', '.join(unexpected)}"

        python_types = {
            "string": str,
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for name, value in arguments.items():
            expected_name = properties[name].get("type")
            expected = python_types.get(expected_name)
            type_matches = expected is None or isinstance(value, expected)
            if expected_name == "integer" and isinstance(value, bool):
                type_matches = False
            if not type_matches:
                return f"Argument '{name}' must have type {expected_name}"

        for name in self.repository_path_args:
            path = arguments.get(name)
            if path is None:
                continue
            if not path.strip():
                return f"Argument '{name}' must not be empty"
            parts = [part for part in path.split("/") if part not in {"", "."}]
            if path.startswith("/"):
                return (
                    f"Argument '{name}' must be repository-relative; "
                    "do not prefix it with /testbed"
                )
            if ".." in parts or ".git" in parts:
                return f"Argument '{name}' must stay inside the task repository"
        return None

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {},
            },
        }

    def get_prompt_description(self) -> str:
        """Возвращает описание инструмента в виде промпта."""

        tool_schema = self.get_schema()

        func = tool_schema["function"]
        name = func["name"]
        description = func["description"]
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        params_block = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            is_required = param_name in required
            req_mark = "REQUIRED" if is_required else "optional"

            params_block.append(
                f"      - {param_name} ({param_type}, {req_mark}): {param_desc}"
            )

        params_str = "\n".join(params_block) if params_block else "      No parameters"

        return f"  - {name}: {description}\n" f"    Parameters:\n" f"{params_str}"
