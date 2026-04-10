from typing import Dict, Any

from langchain_experimental.utilities import PythonREPL

from src.tools.base_tool import BaseTool


class PythonReplTool(BaseTool):

    def __init__(
        self,
        name: str = "python_repl",
        description: str = "Use it to compile Python code.",
    ):
        super().__init__(name=name, description=description)

    def get_schema(self) -> Dict[str, Any]:
        """Возвращает описание инструмента в формате OpenAI Tools API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The python code to execute.",
                        }
                    },
                    "required": ["code"],
                },
            },
        }

    def __call__(self, code: str) -> str:
        """Выполняет переданный Python-код.
        
        Чтобы увидеть результат вычисления, его необходимо явно вывести 
        с помощью функции `print(...)`. Результат выполнения будет доступен пользователю.
        """
        
        repl = PythonREPL()
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Successfully executed:\n : {result}"

