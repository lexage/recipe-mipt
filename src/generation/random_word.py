import random

from openai import OpenAI
from typing import List, Optional
from tqdm import tqdm

from src.agent_constructor.generator import Generator
from src.agent_constructor.core import Text, Document
from src.utils import DOCUMENT_SRC_EXAMPLES

words: List = [
    # Nouns (34)
    'algorithm', 'application', 'array', 'bug', 'class', 'code', 'collection',
    'compiler', 'component', 'data', 'database', 'dependency', 'development',
    'documentation', 'error', 'exception', 'framework', 'function',
    'implementation', 'interface', 'library', 'logic', 'loop', 'method',
    'module', 'object', 'operation', 'package', 'parameter', 'pattern',
    'process', 'script', 'syntax', 'variable',

    # Verbs (33)
    'analyze', 'build', 'calculate', 'call', 'compile', 'compute', 'configure',
    'connect', 'convert', 'create', 'debug', 'deploy', 'design', 'develop',
    'encrypt', 'execute', 'export', 'filter', 'generate', 'handle', 'import',
    'initialize', 'install', 'integrate', 'iterate', 'optimize', 'parse',
    'process', 'read', 'refactor', 'solve', 'transform', 'validate',

    # Adjectives (33)
    'asynchronous', 'clean', 'concurrent', 'configurable', 'correct',
    'efficient', 'encrypted', 'error-free', 'extensible', 'fast', 'flexible',
    'functional', 'generic', 'idempotent', 'immutable', 'iterative',
    'maintainable', 'modular', 'optimal', 'parallel', 'portable', 'readable',
    'recursive', 'reliable', 'responsive', 'robust', 'scalable', 'secure',
    'serializable', 'simple', 'stable', 'testable', 'thread-safe'
]


class RandomWordGenerator(Generator):
    def __init__(
                self,
                url: Optional[str] = None,
                model_name: Optional[str] = None,
                prob: float = 0.01
    ):
        super().__init__("random_word_generator")

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.words = words
        self.prob = max(0.0, min(1.0, prob))
        self.model_name = model_name

    def _get_random_items(
                        self,
                        items: List,
                        min_count: int = 3,
                        max_count: int = 5,
                         ) -> List[str]:

        max_possible = min(max_count, len(items))
        min_possible = min(min_count, len(items))

        count = random.randint(min_possible, max_possible)
        return random.sample(items, count)

    def _run_model(self, example: Text) -> Text:

        random_words = self._get_random_items(self.words)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content":
                    "You are a professional prompt-engineer"},
                {"role": "user", "content": f"""
                Your task is to come up with a programming problem.
                Write this problem as a prompt for a large language model.
                In the prompt text, be sure to use these words: {random_words}.
                Your answer should contain only the prompt, nothing else.
                """},
            ],
            temperature=0.6,
            max_tokens=500
        )

        random_promt: Text = response.choices[0].message.content

        new_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content":
                    "You are professional python-coder"},
                {"role": "user", "content": random_promt +
                    f"""
        To solve the problem pay attention to the examples: {example}.
        In your response, include only the Python code and nothing else.
        Remember to keep the code as simple as possible
        and avoid overcomplicating it where possible.
        The simpler the code, the better.
        Your response should contain nothing but code.
        VERY IMPORTANT: Do not write anything in the answer except the code,
        it should not contain any text, only code.
                    """}
            ],
            temperature=0.1,
            max_tokens=500
        )
        return new_response.choices[0].message.content

    def generate(self, documents: List[Document]) -> List[Document]:
        synth_docs = []
        ids_offset = len(documents)+1

        documents = [
            doc for doc in documents if doc.source == DOCUMENT_SRC_EXAMPLES
            ]
        documents = random.sample(documents, round(len(documents) * self.prob))

        for doc in tqdm(documents, desc="Random Words Generation"):
            synth_docs.append(
                Document(
                    id=str(len(synth_docs) + ids_offset),
                    text=self._run_model(doc.text),
                    source=self.name,
                    metadata={"generated_from": doc.id})
            )

        return synth_docs
