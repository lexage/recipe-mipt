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

topics = [
    "Creating arrays",
    "Loading data from files",
    "Selecting array elements",
    "Changing array shape",
    "Combining arrays",
    "Basic math operations",
    "Finding min and max",
    "Calculating sum and mean",
    "Working with matrices",
    "Generating random numbers",
    "Checking for zeros and empty values",
    "Changing data types",
    "Copying arrays",
    "Removing unnecessary data",
    "Saving arrays to files"
]


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

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a programming example complexity enhancer."},
                {"role": "user", "content": f"""Increase complexity of this example: 
                {task}
                
                You have been given an excerpt from the documentation for the Python library with code example."
                Please increase the complexity of the given programming example a bit.
                You can increase the complexity using, but not limited to, the following methods:
                   - Add new constraints and requirements to the original problem, adding approximately 10 additional words.
                   - Replace a commonly used requirement in the programming task with a less common and more specific one.
                   - If the original problem can be solved with only a few logical steps, please add more reasoning steps.
                   - Provide a piece of erroneous code as a reference to increase misdirection.
                   - Propose higher time or space complexity requirements, but please refrain from doing so frequently
                 
                The result should only contain new example and it`s breif description.
                """}
            ],
            temperature=0.1,
            max_tokens=500
        )

        return response.choices[0].message.content



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

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are incorret programming examples generator"},
                {"role": "user", "content": f"""You have been given an excerpt from the documentation for the Python library with code example:
                {task}

                Please generate incorrect version of this programming example that include one or more semantic bugs." 
                Place the delimiter ```python before every solution example you’ll generate and ``` at the end of the solution code to help me extract just the generated code." 
                Importantly, it should be possible to compile the incorrect solutions and it should be possible to run unit tests for the code.

                As a result return just new example. 
                """}
            ],
            temperature=0.1
        )

        return response.choices[0].message.content


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
                {"role": "system", "content": "You are search query generator"},
                {"role": "user", "content": f"""Given original query: `{task}`, generate {self.options} variations of rewriting original query
                
                Yours output template (only options, nothing else!!!):
                - "option 1"
                - "option 2"
                .
                .
                .
                - "option {self.options}"
                """}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.split('\n')


class RandomWordGenerator(Agent):
    def __init__(self, url: str, model_name: str):
        super().__init__("random_word_generator")
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

    def run(self, query: Text, example: Text) -> Text:
        if self.dummy_mode:
            return query  # , context

        random_words = self.get_random_items(self.words)
        # print(random_words)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional task rephraser"},
                {"role": "user", "content": f"""
                You are given a task: {query}.
                Rephrase it using the given set of words: {random_words}.
                Your paraphrasing should result in a new problem text. 
                It's crucial that the meaning of the problem remains the same. 
                In your answer, include only the paraphrased text, nothing else.
                """},
            ],
            temperature=0.6,
            max_tokens=500
        )

        rephrased_promt: Text = response.choices[0].message.content

        # print(rephrased_promt)

        new_response = self.client.chat.completions.create(
            model=self.model_name,
           messages=[
                {"role": "system", "content": "You are professional python-coder"},
                {"role": "user", "content": rephrased_promt +
                    f"""
                        To solve the problem pay attention to the examples: {example}.
                        In your response, include only the Python code and nothing else.
                        Remember to keep the code as simple as possible and avoid overcomplicating it where possible.
                        The simpler the code, the better. Your response should contain nothing but code.
                        VERY IMPORTANT: Do not write anything in the answer except the code, it should not contain any text, only code.
                    """}
            ],
            temperature=0.1,
            max_tokens=500
        )
        return new_response.choices[0].message.content


class ZeroFewShotGenerator(Agent):
    def __init__(self, url: str, model_name: str):
        super().__init__("zero_few_shot_generator")
        self.dummy_mode = not (url and model_name)
        self.topics = topics
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, query: Text, example: Text) -> tuple[Text, Text, Text]:
        if self.dummy_mode:
            return query

        zero_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional Python developer"},
                {"role": "user", "content": f"""
                    You have a task: {query}.
                    Solve it like an expert with 10 years of experience.
                    In your response, include only the Python code and nothing else.
                    Remember to keep the code as simple as possible and avoid overcomplicating it where possible.
                    The simpler the code, the better. Your response should contain nothing but code.
                    VERY IMPORTANT: Do not write anything in the answer except the code, it should not contain any text, only code.
                """}
                ],
            temperature=0.1,
        )
        zero_example: Text = zero_response.choices[0].message.content

        selected_topic = random.choice(self.topics)
        # print(f"selected_topic - {selected_topic}")
        topic_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional Python developer"},
                {"role": "user", "content": f"""
                    Write a code example on this topic: {selected_topic}
                    In your response, include only the Python code and nothing else.
                    Remember to keep the code as simple as possible and avoid overcomplicating it where possible.
                    The simpler the code, the better. Your response should contain nothing but code.
                    VERY IMPORTANT: Do not write anything in the answer except the code, it should not contain any text, only code.
                """}
            ],
            temperature=0.1,
        )
        topic_example: Text = topic_response.choices[0].message.content

        few_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional Python developer"},
                {"role": "user", "content": f"""
                    You have a task: {query}.
                    Take this as an example: {example}
                    Write another code example that is similar in structure or content to the ones provided to you.
                """}
            ],
            temperature=0.1,
        )
        few_example: Text = few_response.choices[0].message.content

        return zero_example, topic_example, few_example


class InstuctGenerator(Agent):
    def __init__(self, url: str, model_name: str):
        super().__init__("instuct_generator")
        self.dummy_mode = not (url and model_name)
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, example: Text) -> Text:
        if self.dummy_mode:
            return example

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional Python developer and Prompt-engineer"},
                {"role": "user", "content": f"""
                Create a prompt for a large language model that will generate a solution like this: {example}
                The response should contain only the prompt you created and nothing else.
                For example:
                Your input:
                ```python
                    Z = np.random.random((3,3,3))
                    print(Z)
                ```
                Your output:
                Generate Python code that creates a 3D NumPy array with shape (3,3,3) filled with random floats between 0 and 1. 
                Then print the array. Output only the code, no explanations.
                """}
            ],
            temperature=0.1,
        )
        instruction: str = response.choices[0].message.content

        return instruction
