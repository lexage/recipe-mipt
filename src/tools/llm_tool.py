from typing import Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI

from src.tools.base_tool import BaseTool


class LLMArgs(BaseModel):
    query: str = Field(..., description="The text of the llm request.")


class LLMTool(BaseTool):

    arg_schema = LLMArgs
    
    def __init__(
        self,
        name: str = "llm",
        description: str = "A tool for invoking llm. Use llm[<query>].",
        url: str = None,
        model_name: str = None,
        temperature: float = 0.0,
    ):
        super().__init__(name=name, description=description)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature

    def get_schema(self) -> Dict[str, Any]:
        """Возвращает описание инструмента в формате OpenAI Tools API."""
        # Автоматическая генерация JSON Schema из Pydantic модели
        schema = self.arg_schema.model_json_schema()
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def llm(self, prompt: str) -> str:
        """Вызов LLM"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def __call__(self, query: str) -> str:
        return self.llm(query)