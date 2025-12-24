from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text, Document


class CodeEvalGenerator(Agent):
    def __init__(self, url: str = None, model_name: str = None):
        super().__init__("code_eval_generator")
        self.dummy_mode = not (url and model_name)

        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:
        if self.dummy_mode:
            return task

        response = self.client.completions.create(
            model=self.model_name,
            prompt=(
                "You have been given an excerpt from the documentation for the Python library with code example."
                "Please increase the difficulty of the given programming example a bit.\n\n"
                "You can increase the difficulty using, but not limited to, the following methods:\n"
                "   - Add new constraints and requirements to the original problem, adding approximately 10 additional words.\n"
                "   - Replace a commonly used requirement in the programming task with a less common and more specific one.\n"
                "   - If the original problem can be solved with only a few logical steps, please add more reasoning steps.\n"
                "   - Provide a piece of erroneous code as a reference to increase misdirection.\n"
                "   - Propose higher time or space complexity requirements, but please refrain from doing so frequently.\n\n"
                "Example:\n\n"
                )
                + task,
            temperature=0.5,
        )

        return response.choices[0].text


class IncorrectExampleGenerator(Agent):
    def __init__(self, url: str, model_name: str):
        super().__init__("incorrect_example_generator")
        self.dummy_mode = not (url and model_name)

        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:
        if self.dummy_mode:
            return task

        response = self.client.completions.create(
            model=self.model_name,
            prompt=(
                "You have been given an excerpt from the documentation for the Python library with code example.\n"
                "Please generate incorrect version of this programming example that include one or more semantic bugs." 
                "Place the delimiter ```python before every solution example you’ll generate and ``` at the end of the solution code to help me extract just the generated code." 
                "Importantly, it should be possible to compile the incorrect solutions and it should be possible to run unit tests for the code. \n\n"
                "Example:\n\n"
                )
                + task,
            temperature=0.7,
        )

        return response.choices[0].text


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

        response = self.client.completions.create(
            model=self.model_name,
            prompt=(
                "You have been given a search query."
                f"Write down {self.options} options for rephrasing this query: "
                )
                + task,
            temperature=0.7,
        )

        return response.choices[0].text.split('\n')
