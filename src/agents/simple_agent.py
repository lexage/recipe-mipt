from src.agent_constructor.agent import Agent
from openai import OpenAI
from src.agent_constructor.core import Text
from src.pipelines.registry import ComponentRegistry
from src.pipelines.constants import ComponentNames


@ComponentRegistry.register_component(ComponentNames.SIMPLE_AGENT)
class SimpleAgent(Agent):
    def __init__(self, url: str = None, model_name: str = None):
        super().__init__("simple_agent")
        self.dummy_mode = not (url and model_name)
        
        if not self.dummy_mode:
            self.client = OpenAI(
                base_url=url,
                api_key="vllm"
            )

        self.model_name = model_name

    def run(self, context: Text, task: Text) -> Text:

        if self.dummy_mode:
            return f"Answer on {task} using context:\n\n{context}"

        user_prompt = f"""Using context: \n{context}\n\nAnswer on qestion: {task}"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": user_prompt}],
            temperature=0,
        )

        return response.choices[0].message.content
    