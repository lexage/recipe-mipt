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
                {"role": "system", "content":
                    "You are search query generator. Using the task description, generate the required number of search queries to obtain information that will help solve the problem."},
                {"role": "user", "content": f"""[TASK DESCRIPTION]:
                {task}

                [REQUIRED NUMBER OF SEARCH QUERIES]: {self.options}

                [OUTPUT FORMAT]:
                Yours output template (only options, nothing else!!!):
                - <query 1>
                - <query 2
                .
                .
                .
                - <query {self.options}>
                """}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.split('\n')
