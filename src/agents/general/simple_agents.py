from src.agent_constructor.agent import Agent
from openai import OpenAI
from src.agent_constructor.core import Text


class DummyAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
    
    def run(self, task: Text, *args, **kwargs):
        return f"Dummy answer on query:\n'''{task}'''\n"


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

    def run(self, task: Text, context: Text) -> Text:

        if self.dummy_mode:
            return f"Answer on {task} using context:\n\n{context}"

        prompt = f"[CONTEXT]:\n{context}\n[TASK]:\n{task}"

        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=0.2,
            top_p=0.95,
            max_tokens=1024,
            stop=["</code>", "# SOLUTION END"],
        )

        return response.choices[0].text
