from openai import OpenAI
from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text, Document
from typing import List
import random

words: List = [
    # Nouns (34)
    'algorithm', 'application', 'array', 'bug', 'class', 'code', 'collection',
    'compiler', 'component', 'data', 'database', 'dependency', 'development',
    'documentation', 'error', 'exception', 'framework', 'function', 'implementation',
    'interface', 'library', 'logic', 'loop', 'method', 'module', 'object',
    'operation', 'package', 'parameter', 'pattern', 'process', 'script',
    'syntax', 'variable',

    # Verbs (33)
    'analyze', 'build', 'calculate', 'call', 'compile', 'compute', 'configure',
    'connect', 'convert', 'create', 'debug', 'deploy', 'design', 'develop',
    'encrypt', 'execute', 'export', 'filter', 'generate', 'handle', 'import',
    'initialize', 'install', 'integrate', 'iterate', 'optimize', 'parse',
    'process', 'read', 'refactor', 'solve', 'transform', 'validate',

    # Adjectives (33)
    'asynchronous', 'clean', 'concurrent', 'configurable', 'correct', 'efficient',
    'encrypted', 'error-free', 'extensible', 'fast', 'flexible', 'functional',
    'generic', 'idempotent', 'immutable', 'iterative', 'maintainable', 'modular',
    'optimal', 'parallel', 'portable', 'readable', 'recursive', 'reliable',
    'responsive', 'robust', 'scalable', 'secure', 'serializable', 'simple',
    'stable', 'testable', 'thread-safe'
]

# topics = [
#     "DataFrame creation",
#     "CSV/Excel reading",
#     "Row filtering",
#     "Column selection",
#     "Data sorting",
#     "Grouping data",
#     "Aggregation (sum, mean, count)",
#     "Merging tables",
#     "Pivot tables",
#     "Handling NaN",
#     "Working with dates",
#     "Column renaming",
#     "Adding new columns",
#     "Removing duplicates",
#     "Saving to file"
# ]


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


class RandomWordGenerator(Agent):
    def __init__(self, url: str = None, model_name: str = None):
        super().__init__("code_eval_generator")
        self.dummy_mode = not (url and model_name)
        self.words = words
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def get_random_items(self,
                         items: List,
                         min_count: int = 3,
                         max_count: int = 5,
                         ) -> List[str]:

        max_possible = min(max_count, len(items))
        min_possible = min(min_count, len(items))

        count = random.randint(min_possible, max_possible)
        return random.sample(items, count)

    def run(self, query: Text, context: Text) -> Text:
        if self.dummy_mode:
            return query  # , context

        random_words = self.get_random_items(self.words)

        response = self.client.completions.create(
            model=self.model_name,
            prompt=(f"""
                    You are given a task: {query}.
                    Rephrase it using the given set of words: {random_words}.
                    VERY IMPORTANT: Your problem statement should be achievable using this information: {context}
            """
                    ),
            temperature=0.7,
        )
        rephrased_promt: str = response.choices[0].text

        new_response = self.client.completions.create(
            model=self.model_name,
            prompt=(rephrased_promt + f"""
                    To solve the problem, be sure to use the context, pay attention to the examples in it: {context}
            """
                    ),
            temperature=0.3,
        )
        return new_response.choices[0].text


class ZeroFewShotGenerator(Agent):
    def __init__(self, url: str = None, model_name: str = None):
        super().__init__("code_eval_generator")
        self.dummy_mode = not (url and model_name)
        # self.topics = topics
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, query: Text, context: Text, topic: Text) -> Text:
        if self.dummy_mode:
            return query  # , context

        zero_response = self.client.completions.create(
            model=self.model_name,
            prompt=(f"""
                    You are given a task: {query}.
                    Solve this task like an expert with 10 years of experience
            """
                    ),
            temperature=0.1,
        )
        zero_example: str = zero_response.choices[0].text

        topic_response = self.client.completions.create(
            model=self.model_name,
            prompt=(f"""
                    You are given a task: {query}.
                    Write a code example on this topic: {topic}
            """
                    ),
            temperature=0.1,
        )
        topic_example: str = topic_response.choices[0].text

        few_response = self.client.completions.create(
            model=self.model_name,
            prompt=(f"""
                    You are given a task: {query}.
                    Take some code examples from: {context}
                    Write another code example that is similar in structure or content to the ones provided to you.
            """
                    ),
            temperature=0.1,
        )
        few_example: str = few_response.choices[0].text

        return zero_example, topic_example, few_example


class InstuctGenerator(Agent):
    def __init__(self, url: str = None, model_name: str = None):
        super().__init__("code_eval_generator")
        self.dummy_mode = not (url and model_name)
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, query: Text, context: Text) -> Text:
        if self.dummy_mode:
            return query  # , context

        response = self.client.completions.create(
            model=self.model_name,
            prompt=(f"""
                    Find one example of code in {context}.
                    For this example create an instruction for LLM
            """
                    ),
            temperature=0.1,
        )
        instruction: str = response.choices[0].text

        return instruction
