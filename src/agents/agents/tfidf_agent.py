import pickle
import os

from src.agents.agent_constructor.pipeline import Agent
from sklearn.feature_extraction.text import TfidfVectorizer


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
    
    def fit(self, texts):
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit(texts)
            self.save_vectorizer()
    
    def run(self, input):
        return self.vectorizer.transform(input).toarray()

