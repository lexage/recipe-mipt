from src.utils import DOCUMENT_SRC_DOCUMENTS
import random

from openai import OpenAI
from typing import List, Optional
from tqdm import tqdm

from src.agent_constructor.generator import Generator
from src.agent_constructor.core import Text, Document


class ZeroShotGenerator(Generator):
    def __init__(
                self,
                url: Optional[str] = None,
                model_name: Optional[str] = None,
                prob: float = 0.01
                ):
        super().__init__("zero_shot_generator")

        self.client = OpenAI(base_url=url, api_key="vllm")

        self.prob = max(0.0, min(1.0, prob))
        self.model_name = model_name

    def generate(self, documents: List[Document]) -> List[Document]:
        synth_docs = []
        ids_offset = len(documents)+1

        documents = [
            doc for doc in documents if doc.source == DOCUMENT_SRC_DOCUMENTS
            ]
        documents = random.sample(documents, round(len(documents) * self.prob))

        for doc in tqdm(documents, desc="Zero Shot Generation"):
            synth_docs.append(
                Document(
                    id=str(len(synth_docs) + ids_offset),
                    text=self._run_model(),
                    source=self.name,
                    metadata={"generated_from": doc.id})
            )

        return synth_docs

    def _run_model(self) -> Text:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content":
                    "You are a professional Python developer"},
                {"role": "user", "content": """
                    Create python documentation about any library.
                """}
                ],
            temperature=0.6,
        )

        return response.choices[0].message.content
