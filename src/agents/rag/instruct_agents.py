from src.agent_constructor.agent import Agent
from openai import OpenAI
from src.agent_constructor.core import Text


class InstructRationalityAgent(Agent):
    def __init__(
            self, url: str, model_name: str, 
            temperature: float = 0.2,
            top_p: float = 0.95,
            max_tokens: float = 1024,
            stop_tokens: list = []
            ):
        
        super().__init__("instruct_rationality_agent")
        
        self.client = OpenAI(
            base_url=url,
            api_key="vllm"
        )

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens


    def run(self, task: Text, context: Text) -> Text:

        prompt = f"""Read the following documents relevant to the given task:
[TASK] 
{task}

[DOCUMENT]
{context}

Please identify wether documnet is useful to answer the given task or no, and explain how the
contents lead to the answer or why it is non relevant.
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
            ,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop_tokens
        )

        return response.choices[0].message.content
