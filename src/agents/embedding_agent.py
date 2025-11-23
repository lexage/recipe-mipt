from openai import OpenAI
from typing import List

from src.agent_constructor.core import Text
from src.agent_constructor.agent import Agent
from src.pipelines.registry import register_component
from src.pipelines.constants import ComponentNames


@register_component(ComponentNames.EMBEDDING)
class EmbeddigAgent(Agent):
    def __init__(self, url: str = None, model_name: str = None):
        super().__init__("embedding_agent")
        
        self.client = OpenAI(
            base_url=url,
            api_key="vllm"
        )

        self.model_name = model_name

    def run(self, data: Text | List[Text]) -> Text:

        results = self.client.embeddings.create(
            input=data,
            model=self.model_name,
        )

        return [item.embedding for item in results.data]
