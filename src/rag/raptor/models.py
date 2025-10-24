import pickle
import os
import numpy as np

from src.rag.raptor.core import BaseEmbeddingModel, BaseSummarizationModel, BaseQAModel
from sklearn.feature_extraction.text import TfidfVectorizer


class DummyQAModel(BaseQAModel):
    def __init__(self):
        super().__init__()
    
    def answer_question(self, context, question):
        return f"Answer: {question}"


class DummySummarization(BaseSummarizationModel):
    def __init__(self):
        super().__init__()

    def summarize(self, context, max_tokens=150):
        return "SUMMARIZATION"


class DummyEmbedding(BaseEmbeddingModel):
    def __init__(self, size, seed = 333):
        super().__init__()
        self.size = size
        np.random.seed(seed=seed)
    
    def create_embedding(self, text):
        return np.random.random(size=self.size)

class TFIDFEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, vectorizer_path='tfidf_vectorizer.pkl'):
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
    
    def create_embedding(self, input):
        return self.vectorizer.transform(input).toarray()
    