import sys
import joblib
from pathlib import Path
from typing import List, Tuple, Optional
import os
import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter
from src.filtering.textbooks_are_all_you_need.src.utils import (
    load_or_create_embeddings,
    filter_high_quality_documents,
)
from src.filtering.textbooks_are_all_you_need.src.random_forest_cls import (
    RandomForestTrainer,
)
from src.filtering.textbooks_are_all_you_need.src.embeddings import Embedder
from src.filtering.textbooks_are_all_you_need.src.label import EducationalEvaluator


class EducationValueClassifierFilter(Filter):
    """
    Filter for classifying and filtering chunks based on educational value.
    
    This filter implements a three-stage pipeline:
    1. Data preparation: Generates or loads embeddings for all chunks
    2. Model training: Loads a pre-trained model or trains a new classifier on a subsample
    3. Filtering: Applies the trained model to filter chunks, returning only high-quality educational content
    
    The filter automatically manages directory creation for embeddings, annotations, and model storage.
    If a trained model already exists at the specified path, it skips training and uses the existing model.
    """
    
    def __init__(
        self,
        subsample_size: int = 1000,
        api_url: str = "http://shtraukh_vllm:8000/v1",
        model_name: str = "unsloth/gemma-3-12b-it",
        embeddings_path: str = "data/embeddings.npy",
        annotations_path: str = "data/annotations.joblib",
        models_path: str = "data/models",
        limit_labels: int = 20
    ) -> None:
        """
        Initialize the education value classifier filter.
        
        Args:
            subsample_size: Number of chunks to use for annotation and model training
                when no pre-trained model exists. Default is 1000 chunks.
            api_url: URL for the educational value annotation API
            model_name: Name of the model to use for educational value annotation
            embeddings_path: Path for storing/loading chunk embeddings
            annotations_path: Path for storing/loading chunk annotations
            models_path: Path for storing/loading trained classification models
        """
        self.subsample_size = subsample_size
        self.api_url = api_url
        self.model_name = model_name
        self.embeddings_path = embeddings_path
        self.annotations_path = annotations_path
        self.models_path = models_path
        self.limit_labels = limit_labels
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.models_path), exist_ok=True)
    
    def _prepare_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Prepare or load embeddings for all chunks."""
        print("Preparing embeddings for chunks...")
        embedder = Embedder()
        return load_or_create_embeddings(
            chunks, 
            self.embeddings_path, 
            embedder
        )
    
    def _create_subsample(
        self, 
        chunks: List[Chunk], 
        all_embeddings: np.ndarray
    ) -> Tuple[List[Chunk], np.ndarray]:
        """Create a random subsample of chunks for training."""
        if len(chunks) > self.subsample_size:
            print(f"Creating subsample of {self.subsample_size} chunks for training...")
            subsample_indices = np.random.choice(
                len(chunks), size=self.subsample_size, replace=False
            )
            subsample_chunks = [chunks[i] for i in subsample_indices]
            subsample_embeddings = all_embeddings[subsample_indices]
            return subsample_chunks, subsample_embeddings
        else:
            print("Using all chunks for training (chunk count <= subsample size)")
            return chunks, all_embeddings
        
    def save_annotations(self, annotations: List[Optional[int]], path: Optional[str] = None) -> None:
        """
        Save annotations to disk in joblib format.
        
        Args:
            annotations: List of annotations (0/1 or None) for chunks
            path: Custom path to save annotations (defaults to self.annotations_path)
        """
        if path is None:
            path = self.annotations_path
        
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        joblib.dump(annotations, path)
        print(f"Annotations saved to {path}")
    
    def _get_balanced_annotations(
        self, 
        subsample_chunks: List[Chunk], 
        subsample_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        print("Getting balanced annotations (10 per class)...")
        
        annotator = EducationalEvaluator(
            api_url=self.api_url, 
            model_name=self.model_name,
            limit=self.limit_labels
        )
        
        all_labels = annotator.get_annotations(subsample_chunks)
        
        self.save_annotations(all_labels)
        
        labeled_indices = []
        class_counts = {0: 0, 1: 0}
        
        for i, label in enumerate(all_labels):
            if label is not None:
                if (label == 0 and class_counts[0] < 10) or (label == 1 and class_counts[1] < 10):
                    labeled_indices.append(i)
                    class_counts[label] += 1
                elif class_counts[0] >= 10 and class_counts[1] >= 10:
                    break
        
        if labeled_indices:
            filtered_embeddings = subsample_embeddings[labeled_indices]
            filtered_labels = np.array([all_labels[i] for i in labeled_indices])
            
            print(f"Collected {class_counts[0]} class 0 and {class_counts[1]} class 1 examples")
            return filtered_embeddings, filtered_labels
        else:
            print("No valid annotations collected")
            return np.array([]), np.array([])
        
    def _load_existing_model(self, trainer: RandomForestTrainer) -> Optional[object]:
        """Load existing model if available."""
        if os.path.exists(self.models_path):
            print("Loading existing trained model...")
            return trainer.load_model(self.models_path)
        return None
    
    def _train_new_model(
        self, 
        subsample_embeddings: np.ndarray, 
        labels: np.ndarray
    ) -> object:
        """Train a new model on the subsample data."""
        print("Training new model on subsample data...")
        trainer = RandomForestTrainer(model_path=self.models_path)
        train_result = trainer.train(subsample_embeddings, labels)
        return train_result[0]
    
    def _get_or_train_model(
        self, 
        all_embeddings: np.ndarray, 
        chunks: List[Chunk]
    ) -> object:
        trainer = RandomForestTrainer(model_path=self.models_path)
        
        model = self._load_existing_model(trainer)
        if model is not None:
            return model
        
        print("No trained model found. Starting training process...")
        
        subsample_chunks, subsample_embeddings = self._create_subsample(
            chunks, all_embeddings
        )
        
        filtered_embeddings, filtered_labels = self._get_balanced_annotations(
            subsample_chunks, subsample_embeddings
        )
        
        if len(filtered_labels) < 20:
            raise ValueError(
                f"Could not collect enough balanced examples. "
                f"Got {len(filtered_labels)} total, need at least 20"
            )
        
        return self._train_new_model(filtered_embeddings, filtered_labels)
    
    def _filter_chunks(
        self, 
        model: object, 
        all_embeddings: np.ndarray, 
        chunks: List[Chunk]
    ) -> List[Chunk]:
        """Filter chunks using the trained model."""
        print("Applying model to filter chunks...")
        
        # Get predictions for all chunks
        all_labels = model.predict(all_embeddings)
        print(f"Prediction distribution: {np.bincount(all_labels.astype(int))}")
        
        # Filter high-quality chunks
        return filter_high_quality_documents(chunks, all_labels)
    
    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Apply the education value classification filter to chunks.
        
        Processes chunks through the three-stage pipeline and returns only
        chunks classified as high-quality educational content (label "1").
        
        Args:
            chunks: List of Chunk objects to be processed and filtered
            
        Returns:
            List of Chunk objects that passed the quality filter
        """
        if not chunks:
            print("No chunks provided for filtering")
            return []
        
        print("=== Stage 1: Data Preparation ===")
        all_embeddings = self._prepare_embeddings(chunks)
        
        print("=== Stage 2: Model Training ===")
        model = self._get_or_train_model(all_embeddings, chunks)
        
        print("=== Stage 3: Data Filtering ===")
        high_quality_chunks = self._filter_chunks(model, all_embeddings, chunks)
        
        print(
            f"Filtered {len(high_quality_chunks)} high-quality chunks "
            f"out of {len(chunks)} original chunks "
            f"({len(high_quality_chunks)/len(chunks):.1%} retention rate)"
        )

        return high_quality_chunks