import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer


class DummyPlanner:
    def __init__(self, max_subqueries: int):
        self.max_num_queries = max_subqueries
    
    def run(self, query: str):
        return [f"#{i} Subquery of query '{query}'" for i in range(self.max_num_queries)]


class TFIDFEmbeddingFunction:
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
    
    def __call__(self, input):
        return self.vectorizer.transform(input).toarray()
    
    def embed_query(self, input):
        return self.__call__(input)
    
    def name(self):
        return "tf-idf"
