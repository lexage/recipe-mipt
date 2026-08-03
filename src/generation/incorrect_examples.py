import random

from openai import OpenAI
from typing import List, Optional
from tqdm import tqdm
from src.agent_constructor.generator import Generator
from src.agent_constructor.core import Text, Document
from src.utils import DOCUMENT_SRC_EXAMPLES


class IncorrectExampleGenerator(Generator):
    def __init__(
                self,
                url: Optional[str] = None,
                model_name: Optional[str] = None,
                prob: float = 0.01
    ):
        super().__init__("incorrect_examples_generator")

        self.client = OpenAI(base_url=url, api_key="vllm")

        self.prob = max(0.0, min(1.0, prob))
        self.model_name = model_name

    def generate(self, documents: List[Document]) -> List[Document]:
        synth_docs: List[Document] = []
        ids_offset = len(documents)+1

        documents = [
            doc for doc in documents if doc.source == DOCUMENT_SRC_EXAMPLES
            ]
        documents = random.sample(documents, round(len(documents) * self.prob))

        for doc in tqdm(documents, desc="Incorrect Examples Generation"):
            synth_docs.append(
                Document(
                    id=str(len(synth_docs) + ids_offset),
                    text=self._run_model(doc.text),
                    source=self.name,
                    metadata={"generated_from": doc.id})
            )

        return synth_docs

    def _run_model(self, example: Text) -> Text:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content":
                    "You are incorret programming examples generator"},
                {"role": "user", "content":
                    f"""
    You have been given an excerpt from the documentation for the Python library with code example:
    {example}

    Please generate incorrect version of this programming example that include one or more semantic bugs."
    Place the delimiter ```python before every solution example you’ll generate and ``` at the end of the solution code to help me extract just the generated code."
    Importantly, it should be possible to compile the incorrect solutions and it should be possible to run unit tests for the code.

    As a result return just new example.
                """}
            ],
            temperature=0.1
        )

        return response.choices[0].message.content
