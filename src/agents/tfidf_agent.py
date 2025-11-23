import pickle
import os
from typing import List

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from sklearn.feature_extraction.text import TfidfVectorizer
from src.pipelines.registry import register_component
from src.pipelines.constants import ComponentNames


@register_component(ComponentNames.TFIDF_EMBEDDING)
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
