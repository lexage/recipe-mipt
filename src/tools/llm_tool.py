from typing import Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool


class LLMToolArgs(BaseModel):
    query: str = Field(..., description="The text of the llm request.")


class LLMTool(BaseTool):

    def __init__(
        self,
        name: str = "llm",
        description: str = "A tool for invoking llm. Use llm[<query>].",
        url: str = None,
        model_name: str = None,
        temperature: float = 0.0,
    ):
        super().__init__(name=name, description=description)
        self.client = OpenAI(base_url=url, api_key="vllm", timeout=600.0, max_retries=2)
        self.model_name = model_name
        self.temperature = temperature
        self.args = LLMToolArgs

    def get_schema(self) -> Dict[str, Any]:
            """Автоматическая генерация схемы из Pydantic модели."""
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": LLMToolArgs.model_json_schema()
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
