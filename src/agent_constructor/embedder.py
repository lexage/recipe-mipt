"""
Эмбеддер для превращения текста в вектора
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class Embedder:
    """
    Класс для создания эмбеддингов текста.
    Использует sentence-transformers для качественных векторов.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация эмбеддера.
        
        Args:
            model_name: Название модели из sentence-transformers
                       'all-MiniLM-L6-v2' - маленькая и быстрая (384 размера)
                       'all-mpnet-base-v2' - побольше, но точнее (768 размера)
        """
        
        self.model = SentenceTransformer(model_name)
        print(f"Размер эмбеддингов: {self.model.get_sentence_embedding_dimension()}")
    
    def __call__(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Превращает текст в вектор.
        
        Args:
            text: Один текст или список текстов
            
        Returns:
            Вектор (или список векторов)
        """
        if isinstance(text, str):
            return self.model.encode(text).tolist()
        else:
            return self.model.encode(text).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Эмбеддинг для нескольких текстов сразу.
        
        Args:
            texts: Список текстов
            
        Returns:
            Список векторов
        """
        return self.model.encode(texts).tolist()