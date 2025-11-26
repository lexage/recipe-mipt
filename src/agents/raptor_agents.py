from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.pipelines.constants import ComponentNames
from src.pipelines.registry import ComponentRegistry


@ComponentRegistry.register_component(ComponentNames.RAPTOR_QA_AGENT)
class RaptorQAAgent(Agent):
    def __init__(self, url: str = None, model_name: str = None):
        super().__init__("raptor_qa_agent")
        self.dummy_mode = not (url and model_name)

        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, query: Text, context: Text) -> Text:
        if self.dummy_mode:
            return f"Dummy QA answer to '{query}' using context: {context[:50]}..."

        prompt = (
            f"using the following information {context}. "
            f"Answer the following question in less than 5-7 words, if possible: {query}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return response.choices[0].message.content


@ComponentRegistry.register_component(ComponentNames.RAPTOR_SUMM_AGENT)
class RaptorSummarizationAgent(Agent):
    def __init__(self, url: str = None, model_name: str = None):
        super().__init__("raptor_summ_agent")
        self.dummy_mode = not (url and model_name)

        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, context: Text, max_tokens: int = 150) -> Text:
        if self.dummy_mode:
            return f"Summary (<= {max_tokens} tokens) of: {context[:50]}..."

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Write a summary of the following, including as many key "
                        f"details as possible: {context}:"
                    ),
                },
            ],
            max_tokens=max_tokens,
            temperature=0,
        )

        return response.choices[0].message.content

