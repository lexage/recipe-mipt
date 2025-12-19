import pickle
import os

from openai import OpenAI
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

from src.agent_constructor.core import Text
from src.agent_constructor.agent import Agent
from src.pipelines.registry import ComponentRegistry
from src.pipelines.constants import ComponentNames


@ComponentRegistry.register_component(ComponentNames.EMBEDDING_AGENT)
class EmbeddingAgent(Agent):
    def __init__(self, url: str = None, model_name: str = None):
        super().__init__("embedding_agent")
        
        self.client = OpenAI(
            base_url=url,
            api_key="vllm"
        )

        self.model_name = model_name

    def run(self, data: Text | List[Text]) -> Text:

        results = self.client.embeddings.create(
            input=data,
            model=self.model_name,
        )

        return [item.embedding for item in results.data]


@ComponentRegistry.register_component(ComponentNames.TFIDF_EMBEDDING)
class TFIDFEmbedding(Agent):
    def __init__(self, name: str, vectorizer_path:str ='data/tfidf_vectorizer.pkl'):
        super().__init__(name)        
        self.vectorizer_path = vectorizer_path
        self.vectorizer = None 
        self._load_or_init_vectorizer()
    
    def _load_or_init_vectorizer(self):
        if os.path.exists(self.vectorizer_path):
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            self.vectorizer = TfidfVectorizer(max_features=384)
    
    def save_vectorizer(self):
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def fit(self, texts: List[Text]):
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit(texts)
            self.save_vectorizer()
    
    def run(self, data: Text | List[Text]):
        if isinstance(data, Text):
            data = [data]
        return self.vectorizer.transform(data).toarray()
