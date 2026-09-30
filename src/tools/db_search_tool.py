from typing import Dict, Any
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool
from src.agent_constructor.db import IDB
from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.context_engine import Retriever


class DBSearchToolArgs(BaseModel):
    query: str = Field(..., description="The text of the query to be searched in the database.")

class DBSearchTool(BaseTool):

    def __init__(
        self,
        name: str = "db_search",
        description: str = "A tool for extracting information from a database.",
        db: IDB = None,
        top_k: int = 5,
        retriever: Retriever = None,
        context_assembler: ContextAssembler = None,
    ):
        super().__init__(name=name, description=description)
        self.db = db
        self.top_k = top_k
        self.retriever = retriever
        self.context_assembler = context_assembler
        self.args = DBSearchToolArgs

    def get_schema(self) -> Dict[str, Any]:
            """Автоматическая генерация схемы из Pydantic модели."""
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": DBSearchToolArgs.model_json_schema()
                },
            }

    def __call__(self, query: str) -> str:
        """Извлекает контекст из базы данных"""
        if self.retriever and self.context_assembler:
            retrieved = self.retriever.retrieve(query, self.top_k)
            retrieved_context = self.context_assembler.assemble(retrieved)
            return f"query: {query}\n\nretrieved context: {retrieved_context}"
        return "No retriever or context assembler available."