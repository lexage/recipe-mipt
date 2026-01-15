import os
import torch
import numpy as np
from typing import List, Optional, Union
from abc import ABC, abstractmethod
from transformers import RobertaTokenizer, RobertaModel
from openai import OpenAI

from src.agent_constructor.core import Document



class EmbedderBase(ABC):
    """Abstract base class for document embedders.
    
    This class defines the interface that all embedder implementations must follow.
    It provides a standardized way to generate embeddings from documents.
    """
    
    @abstractmethod
    def get_document_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Generate embeddings for a list of documents.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            numpy.ndarray: Array of document embeddings with shape (n_documents, embedding_dim)
            
        Raises:
            NotImplementedError: If the method is not implemented in a subclass
        """
        pass


class TransformersCodeEmbedder(EmbedderBase):
    """Document embedder using Hugging Face transformers library with CodeBERT model.
    
    This embedder uses the Microsoft CodeBERT model to generate embeddings for code documents.
    It processes documents locally on the specified device (CPU or GPU) and extracts the CLS token
    embedding from the model's last hidden state.
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base", device: Optional[str] = None):
        """Initialize the transformers-based code embedder.
        
        Args:
            model_name: Name or path of the pre-trained CodeBERT model to use. 
                       Defaults to "microsoft/codebert-base".
            device: Device to run the model on ("cpu", "cuda", etc.). If None, automatically
                   selects "cuda" if available, otherwise "cpu". Defaults to None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_document_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Generate embeddings for documents using the transformers model.
        
        Args:
            documents: List of Document objects containing text to embed
            
        Returns:
            numpy.ndarray: Array of document embeddings with shape (n_documents, 768)
                          where 768 is the embedding dimension of CodeBERT.
            
        Note:
            - Text is truncated to 512 tokens if longer
            - The CLS token embedding (first token) is used as the document embedding
            - Model runs in evaluation mode with no gradients
        """
        embeddings = []
        for doc in documents:
            inputs = self.tokenizer(
                doc.text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            embeddings.append(cls_embedding)
        
        return np.array(embeddings)


class VLLMOpenAIEmbedder(EmbedderBase):
    """Document embedder using OpenAI-compatible API with vLLM backend.
    
    This embedder sends document text to a vLLM server running an OpenAI-compatible API
    endpoint to generate embeddings. It's useful for offloading embedding computation
    to a dedicated server or using different embedding models hosted by vLLM.
    """
    
    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123"  # dummy key; vLLM usually ignores it unless auth enabled
    ):
        """Initialize the vLLM OpenAI-compatible embedder.
        
        Args:
            model_name: Name of the embedding model to use on the vLLM server
            api_base: Base URL for the OpenAI-compatible API endpoint. Defaults to
                     "http://localhost:8000/v1" (standard vLLM port).
            api_key: API key for authentication. vLLM typically ignores this unless
                    authentication is explicitly enabled. Defaults to "token-abc123".
        """
        self.model_name = model_name
        self.client = OpenAI(base_url=api_base, api_key=api_key)

    def get_document_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Generate embeddings for documents using the vLLM OpenAI API.
        
        Args:
            documents: List of Document objects containing text to embed
            
        Returns:
            numpy.ndarray: Array of document embeddings with shape (n_documents, embedding_dim)
                          where embedding_dim depends on the model used on the vLLM server.
            
        Raises:
            openai.APIError: If the API request fails
            openai.AuthenticationError: If authentication fails (if enabled on server)
            
        Note:
            - This method sends a batch request to the vLLM server
            - All document texts are sent in a single API call
            - The server handles tokenization and embedding generation
        """
        documents_text = [doc.text for doc in documents]
        response = self.client.embeddings.create(
            input=documents_text,
            model=self.model_name
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)


class Embedder:
    """Factory class for creating document embedders with configurable backends.
    
    This class provides a unified interface to different embedding backends (transformers or vLLM).
    It abstracts away the implementation details and allows easy switching between different
    embedding methods through configuration.
    """
    
    def __init__(
        self,
        backend: str = "transformers",  # or "vllm"
        model_name: str = "microsoft/codebert-base",
        device: Optional[str] = None,
        vllm_api_base: str = "http://localhost:8000/v1",
        vllm_api_key: str = "vllm"
    ):
        """Initialize the embedder factory with specified backend and configuration.
        
        Args:
            backend: Embedding backend to use. Options are:
                    - "transformers": Local transformers model (default)
                    - "vllm": Remote vLLM server with OpenAI API
            model_name: Name of the embedding model to use. For transformers backend,
                       this is the Hugging Face model name. For vLLM backend, this is
                       the model name configured on the server.
            device: Device to use for transformers backend ("cpu", "cuda", etc.).
                   Ignored for vLLM backend. If None, auto-detects available devices.
            vllm_api_base: API base URL for vLLM backend. Ignored for transformers backend.
                          Defaults to "http://localhost:8000/v1".
            vllm_api_key: API key for vLLM backend. Ignored for transformers backend.
                         Defaults to "vllm".
            
        Raises:
            ValueError: If backend is not "transformers" or "vllm"
            Exception: If model loading fails for transformers backend
            openai.OpenAIError: If connection to vLLM server fails during initialization
        """
        if backend == "transformers":
            self.embedder = TransformersCodeEmbedder(model_name=model_name, device=device)
        elif backend == "vllm":
            self.embedder = VLLMOpenAIEmbedder(
                model_name=model_name,
                api_base=vllm_api_base,
                api_key=vllm_api_key
            )
        else:
            raise ValueError("backend must be 'transformers' or 'vllm'")

    def get_document_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Generate embeddings for documents using the selected backend.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            numpy.ndarray: Array of document embeddings with shape (n_documents, embedding_dim)
            
        Note:
            This method delegates to the underlying embedder implementation based on
            the backend selected during initialization.
        """
        return self.embedder.get_document_embeddings(documents)