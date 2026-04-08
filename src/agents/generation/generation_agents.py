from openai import OpenAI
from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text


class QueryGenerator(Agent):
    def __init__(self, url: str, model_name: str, options: int = 3):
        super().__init__("query_generator")
        self.dummy_mode = not (url and model_name)
        self.options = options

        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:
        if self.dummy_mode:
            return task

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": f"""You are search query generator. Using the task description, generate the required number of different search queries to obtain information that will help solve the problem.
                [OUTPUT FORMAT]:
                Yours answer will be divided by lines and each line would be considered as search query and will be passed straight to the search engine.
                So yours answer should consist of just {self.options} lines, each one containing search query. Do not write redundant phrases like "Here are search queries to help you solve the problem". 
                I expect a response in this format containing only search queries:
                
                <query 1>
                <query 2>
                ...
                 """},
                {"role": "user", "content": f"""[PROBLEM DESCRIPTION]:
                {task}

                Provide {self.options} search queries!
                """}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.split('\n')
