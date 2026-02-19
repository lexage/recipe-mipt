import random

from openai import OpenAI
from typing import List
from tqdm import tqdm

from src.agent_constructor.generator import Generator
from src.agent_constructor.core import Text, Document
from src.utils import DOCUMENT_SRC_EXAMPLES

class CodeEvalGenerator(Generator):
    def __init__(self, url: str = None, model_name: str = None, prob: float = 0.01):
        super().__init__("code_eval_generator")

        self.client = OpenAI(base_url=url, api_key="vllm")
        
        self.prob = max(0.0, min(1.0, prob))
        self.model_name = model_name

    def generate(self, documents: List[Document]) -> List[Document]:
        synth_docs = []
        ids_offset = len(documents)+1

        documents = [doc for doc in documents if doc.source == DOCUMENT_SRC_EXAMPLES]
        documents = random.sample(documents, round(len(documents) * self.prob))

        for doc in tqdm(documents, desc="Code Eval Generation"):
            synth_docs.append(
                Document(
                    id=str(len(synth_docs) + ids_offset),
                    text=self._run_model(doc.text),
                    source=self.name,
                    metadata={"generated_from" : doc.id})
            )
        
        return synth_docs

    def _run_model(self, example: Text) -> Text:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a programming example complexity enhancer."},
                {"role": "user", "content": f"""Increase complexity of this example: 
                {example}
                
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