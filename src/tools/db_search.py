from typing import Dict, Any

from src.agent_constructor.db import IDB
from src.agent_constructor.context_engine import ContextAssembler
from src.tools.base_tool import BaseTool


class DBSearchTool(BaseTool):

    def __init__(
        self,
        name: str = "db_search",
        description: str = "A tool for extracting information from a database.",
        db: IDB = None,
        top_k: int = 5,
        context_assembler: ContextAssembler = None,
    ):
        super().__init__(name=name, description=description)
        self.db = db
        self.top_k = top_k
        self.context_assembler = context_assembler

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
                        "query": {
                            "type": "string",
                            "description": "The text of the query to be searched in the database.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def __call__(self, query: str) -> str:
        """Извлекает контекст из базы данных"""
        if self.db and self.context_assembler:
            retrieved = self.db.query(query, self.top_k)
            retrieved_context = self.context_assembler.assemble(retrieved)
            return f"query: {query}\n\nretrieved context: {retrieved_context}"
        return "No database or context assembler available."