import random

from openai import OpenAI
from typing import List, Optional
from tqdm import tqdm

from agent_constructor.generator import Generator
from src.agent_constructor.core import Text, Document
from src.utils import DOCUMENT_SRC_DOCUMENTS

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


class RandomTopicGenerator(Generator):
    def __init__(
                self,
                url: Optional[str] = None,
                model_name: Optional[str] = None,
                prob: float = 0.01
    ):
        super().__init__("random_topic_generator")

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.topics = topics
        self.prob = max(0.0, min(1.0, prob))
        self.model_name = model_name

    def _run_model(self, example: Text) -> Text:

        selected_topic = random.choice(self.topics)

        topic_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content":
                    "You are a professional Python developer"},
                {"role": "user", "content": f"""
        Write a python documentation example on this topic: {selected_topic}
        It is very important to write an example on your topic,
        similar to the example: {example}
                """}
            ],
            temperature=0.1,
        )
        topic_example: Text = topic_response.choices[0].message.content

        return topic_example

    def generate(self, documents: List[Document]) -> List[Document]:
        synth_docs = []
        ids_offset = len(documents)+1

        documents = [
            doc for doc in documents if doc.source == DOCUMENT_SRC_DOCUMENTS
            ]
        documents = random.sample(documents, round(len(documents) * self.prob))

        for doc in tqdm(documents, desc="Random Topic Generation"):
            synth_docs.append(
                Document(
                    id=str(len(synth_docs) + ids_offset),
                    text=self._run_model(doc.text),
                    source=self.name,
                    metadata={"generated_from": doc.id})
            )

        return synth_docs
