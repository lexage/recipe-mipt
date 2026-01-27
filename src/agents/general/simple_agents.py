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

class ICVAgent(Agent):
    def __init__(self, url: str = None, model_name: str = None):
        super().__init__("icv_agent")
        
        self.client = OpenAI(
            base_url=url,
            api_key="vllm"
        )

        self.model_name = model_name

    def run(self, icv_xargs: dict, task: Text) -> Text:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": task}],
            temperature=0,
            extra_body={
                "vllm_xargs" : icv_xargs
            }
        )

        return response.choices[0].message.content