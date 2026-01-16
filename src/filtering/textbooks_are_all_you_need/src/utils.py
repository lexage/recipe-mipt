import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from typing import List
import numpy as np
import joblib

from src.agent_constructor.core import Document
from src.filtering.textbooks_are_all_you_need.src.embeddings import Embedder
from src.filtering.textbooks_are_all_you_need.src.label import EducationalEvaluator


def load_or_create_embeddings(
    documents: List[Document], embeddings_path: str, embedder: Embedder
) -> np.ndarray:
    """
    Load embeddings from file or create new ones if the file doesn't exist.

    Args:
        documents: List of Document objects to generate embeddings for
        embeddings_path: Path to the embeddings file

    Returns:
        numpy.ndarray: Array of document embeddings
    """
    if os.path.exists(embeddings_path):
        print(f"Loading embeddings from {embeddings_path}")
        return np.load(embeddings_path)
    else:
        print("Creating embeddings for documents...")
        # Assuming there's a function to get document embeddings

        embeddings = embedder.get_document_embeddings(documents)
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.save(embeddings_path, embeddings)
        return embeddings


def load_or_create_annotations(
    subsample_documents: List[Document],
    annotations_path: str,
    annotator: EducationalEvaluator,
) -> List[int]:
    """
    Load annotations from file or create new ones using LLM if the file doesn't exist.

    Args:
        subsample_documents: List of Document objects to annotate
        annotations_path: Path to the annotations file

    Returns:
        List[int]: List of annotation labels (0 or 1)
    """
    if os.path.exists(annotations_path):
        print(f"Loading annotations from {annotations_path}")
        return joblib.load(annotations_path)
    else:
        print("Getting automatic annotations using LLM...")
        annotations = annotator.get_annotations(subsample_documents)
        os.makedirs(os.path.dirname(annotations_path), exist_ok=True)
        joblib.dump(annotations, annotations_path)
        return annotations


def filter_high_quality_documents(
    documents: List[Document], labels: np.ndarray
) -> List[Document]:
    """
    Return documents with label '1' (high-quality data).

    Args:
        documents: List of Document objects to filter
        labels: Array of labels corresponding to each document (0 or 1)

    Returns:
        List[Document]: Filtered list containing only high-quality documents
    """
    return [doc for doc, label in zip(documents, labels) if label == 1]
