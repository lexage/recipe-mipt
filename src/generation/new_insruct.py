import random

from openai import OpenAI
from typing import List, Optional
from tqdm import tqdm

from src.agent_constructor.generator import Generator
from src.agent_constructor.core import Text, Document
from src.utils import DOCUMENT_SRC_EXAMPLES


class InstructGenerator(Generator):
    def __init__(
                self,
                url: Optional[str] = None,
                model_name: Optional[str] = None,
                prob: float = 0.01
    ):
        super().__init__("instruct_generator")
        self.client = OpenAI(base_url=url, api_key="vllm")

        self.prob = max(0.0, min(1.0, prob))
        self.model_name = model_name

    def _run_model(self, example: Text) -> Text:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content":
                    "You are a professional Prompt-engineer"},
                {"role": "user", "content": f"""
    Create a prompt for a LLM that will generate a solution like this:
    {example}
    The response should contain only the prompt you created and nothing else.
    For example:
    Your input:
    ```python
        Z = np.random.random((3,3,3))
        print(Z)
    ```
    Your output:
    Generate a NumPy array with a random shape filled with random values and then print the resulting array.
                """}
            ],
            temperature=0.6,
        )
        instruction: str = response.choices[0].message.content

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content":
                    "You are a professional Python developer"},
                {"role": "user", "content": instruction}
            ],
            temperature=0.1,
        )
        answer: str = response.choices[0].message.content

        return answer

    def generate(self, documents: List[Document]) -> List[Document]:
        synth_docs: list[Document] = []
        ids_offset = len(documents)+1
        print("a")
        documents = [
            doc for doc in documents if doc.source == DOCUMENT_SRC_EXAMPLES
            ]
        documents = random.sample(documents, round(len(documents) * self.prob))

        for doc in tqdm(documents, desc="Instruct Generation"):
            synth_docs.append(
                Document(
                    id=str(len(synth_docs) + ids_offset),
                    text=self._run_model(doc.text),
                    source=self.name,
                    metadata={"generated_from": doc.id})
            )
            print(f"synth_docs: {synth_docs}")

        return synth_docs
