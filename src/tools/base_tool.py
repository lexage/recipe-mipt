from abc import abstractmethod
from typing import Any, Dict, Optional

from src.agent_constructor.core import Block


class ToolResult(str):
    """String-compatible result carrying generic tool execution status."""

    success: bool
    error_code: Optional[str]
    retryable: bool

    def __new__(
        cls,
        content: str,
        *,
        success: bool = True,
        error_code: Optional[str] = None,
        retryable: bool = True,
    ) -> "ToolResult":
        instance = super().__new__(cls, content)
        instance.success = success
        instance.error_code = error_code
        instance.retryable = retryable
        return instance

    @classmethod
    def ok(cls, content: str) -> "ToolResult":
        return cls(content, success=True)

    @classmethod
    def error(
        cls,
        content: str,
        *,
        error_code: str = "tool_error",
        retryable: bool = True,
    ) -> "ToolResult":
        return cls(
            content,
            success=False,
            error_code=error_code,
            retryable=retryable,
        )


class BaseTool(Block):
    """Абстрактный класс инструмента."""

    def __init__(self, name: str = "", description: str = ""):
        self.name = name
        self.description = description

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