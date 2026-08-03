import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter
from src.agents.general.embedding_agents import EmbeddingAgent
from src.filtering.textbooks_are_all_you_need.src.label import EducationalEvaluator
from src.filtering.textbooks_are_all_you_need.src.random_forest_cls import (
    RandomForestTrainer,
)


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
        llm_url: str = "http://shtraukh_vllm:8000/v1",
        llm_model: str = "unsloth/gemma-3-12b-it",
        embedding_url: str = "http://shtraukh_vllm:8001/v1",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embeddings_path: str = "data/embeddings.npy",
        annotations_path: str = "data/annotations.joblib",
        limit_labels: int = 20,
        models_path: str = "data/models",
        datasets_path: str = "data/datasets.npz",
        metadata_path: str = "data/metadata.json",
        save_metadata: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the educational value classifier filter with configurable paths and model settings.

        This component orchestrates annotation (via LLM), embedding generation, and training
        of a lightweight classifier (e.g., Random Forest) to filter text chunks by educational quality.

        Args:
            subsample_size (int): Number of randomly selected text chunks to annotate and use for training
                when no pre-trained model is available. Default: 1000.
            llm_url (str): Base URL of the LLM inference service (OpenAI-compatible) used for zero-shot
                educational value annotation. Default: "http://shtraukh_vllm:8000/v1".
            llm_model (str): Name of the LLM to use for annotation (e.g., "unsloth/gemma-3-12b-it").
            embedding_url (str): Base URL of the embedding service (OpenAI-compatible) for vectorizing chunks.
                Default: "http://shtraukh_vllm:8001/v1".
            embedding_model (str): Name of the embedding model (e.g., "Qwen/Qwen3-Embedding-0.6B").
            embeddings_path (str): Path to load from or save chunk embeddings in NumPy (.npy) format.
            annotations_path (str): Path to load from or save LLM-generated binary annotations in Joblib format.
            limit_labels (int): Maximum number of annotated samples to retain per class (or total, depending on implementation)
                to control dataset size or enforce class balance. Default: 20.
            models_path (str): Directory to store or load trained classifier models (e.g., joblib files).
            datasets_path (str): Path to store or load the train/test split in NumPy NPZ format.
            metadata_path (str): Path to save training metadata (metrics, config, etc.) in JSON format.
            save_metadata (bool): Whether to persist training metadata (e.g., accuracy, feature importances) to disk.
                Default: False.
            test_size (float): Proportion of the annotated data to reserve for evaluation (between 0.0 and 1.0).
                Default: 0.2.
            random_state (int): Seed for reproducibility in sampling, dataset splitting, and model initialization.
                Default: 42.
        """
        self.subsample_size = subsample_size
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.embedding_url = embedding_url
        self.embedding_model = embedding_model
        self.embeddings_path = embeddings_path
        self.annotations_path = annotations_path
        self.limit_labels = limit_labels
        self.limit_labels_per_class = limit_labels // 2
        self.models_path = models_path
        self.datasets_path = datasets_path
        self.test_size = test_size
        self.random_state = random_state
        self.metadata_path = metadata_path
        self.save_metadata = save_metadata

        os.makedirs(os.path.dirname(self.models_path), exist_ok=True)

    def _prepare_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Prepare or load precomputed embeddings for the given list of chunks.

        This internal method first checks if embeddings have already been saved at
        `self.embeddings_path`. If so, it loads and returns them. Otherwise, it generates
        new embeddings by sending chunk texts to a remote embedding service via
        `EmbeddingAgent`, saves the resulting embeddings to disk for future use,
        and returns them as a NumPy array.

        Args:
            chunks: A list of Chunk objects containing the text content to embed.

        Returns:
            A NumPy array of shape (n_chunks, embedding_dim) containing the embeddings
            corresponding one-to-one with the input chunks.
        """
        print("Preparing embeddings for chunks...")
        embedder = EmbeddingAgent(self.embedding_url, self.embedding_model)

        if os.path.exists(self.embeddings_path):
            print(f"Loading embeddings from {self.embeddings_path}")
            return np.load(self.embeddings_path)
        else:
            print("Creating embeddings for chunks...")
            # Assuming there's a function to get chunk embeddings
            chunks_text = [chunk.text for chunk in chunks]
            embeddings = np.array(embedder.run(chunks_text))
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            np.save(self.embeddings_path, embeddings)
            return embeddings

    def _create_subsample(
        self, chunks: List[Chunk], all_embeddings: np.ndarray
    ) -> Tuple[List[Chunk], np.ndarray]:
        """
        Create a random subsample of chunks and their corresponding embeddings for training.

        If the total number of input chunks exceeds `self.subsample_size`, this method randomly
        selects a subset of that size without replacement. Otherwise, it returns all chunks
        and embeddings unchanged. This helps control computational cost and memory usage
        during model training while maintaining representativeness.

        Args:
            chunks: A list of Chunk objects to potentially subsample.
            all_embeddings: A NumPy array of shape (n_chunks, embedding_dim) containing
                            embeddings corresponding to each chunk.

        Returns:
            A tuple containing:
                - A list of subsampled (or original) Chunk objects.
                - A NumPy array of the corresponding embeddings.
        """
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

    def _save_annotations(
        self, annotations: List[Optional[int]], path: Optional[str] = None
    ) -> None:
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

    def _extract_balanced_subset(
        self, labels: List[Optional[int]], embeddings: np.ndarray, strict: bool = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract a balanced subset of examples (up to self.limit_labels_per_class per class).

        Args:
            labels: List of labels (may contain None values)
            embeddings: Embeddings corresponding to labels
            strict: If True, return None if unable to collect self.limit_labels_per_class examples for each class.
                    If False, return whatever was collected.

        Returns:
            Tuple of (filtered_embeddings, filtered_labels) if examples were collected,
            None if no examples collected or strict=True and target not met.
        """
        labeled_indices = []
        class_counts = {0: 0, 1: 0}

        for i, label in enumerate(labels):
            if label is not None and label in (0, 1):
                if (label == 0 and class_counts[0] < self.limit_labels_per_class) or (
                    label == 1 and class_counts[1] < self.limit_labels_per_class
                ):
                    labeled_indices.append(i)
                    class_counts[label] += 1
                if (
                    class_counts[0] >= self.limit_labels_per_class
                    and class_counts[1] >= self.limit_labels_per_class
                ):
                    break

        # Check if we have any valid examples
        if not labeled_indices:
            if strict:
                print("No valid annotations collected")
            return None

        # In strict mode, require exact balance
        if strict and (
            class_counts[0] < self.limit_labels_per_class
            or class_counts[1] < self.limit_labels_per_class
        ):
            print(
                f"Insufficient balanced examples: "
                f"class 0: {class_counts[0]}, class 1: {class_counts[1]} "
                f"(need {self.limit_labels_per_class} each)"
            )
            return None

        filtered_embeddings = embeddings[labeled_indices]
        filtered_labels = np.array([labels[i] for i in labeled_indices])

        mode = "Loaded" if strict else "Collected"
        print(
            f"{mode} balanced annotations: {class_counts[0]} class 0 and {class_counts[1]} class 1 examples"
        )
        return filtered_embeddings, filtered_labels

    def _load_existing_annotations(
        self, subsample_chunks: List[Chunk], subsample_embeddings: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load existing annotations from disk and extract a balanced subset (10 per class).

        Args:
            subsample_chunks: List of subsampled chunks
            subsample_embeddings: Embeddings corresponding to subsample_chunks

        Returns:
            Tuple of (filtered_embeddings, filtered_labels) if at least 10 examples per class exist,
            otherwise None to trigger new annotation.
        """
        if not os.path.exists(self.annotations_path):
            print(f"No existing annotations found at {self.annotations_path}")
            return None

        try:
            all_labels = joblib.load(self.annotations_path)
            print(
                f"Loaded {len(all_labels)} existing annotations from {self.annotations_path}"
            )

            # Verify annotation count matches subsample size
            if len(all_labels) != len(subsample_chunks):
                print(
                    f"Annotation count ({len(all_labels)}) doesn't match subsample size ({len(subsample_chunks)})"
                )
                print(
                    "Annotations may be from a different dataset version - skipping reuse"
                )
                return None

            # Extract balanced subset with strict requirements
            return self._extract_balanced_subset(
                all_labels, subsample_embeddings, strict=True
            )

        except Exception as e:
            print(f"Error loading annotations: {e}")
            return None

    def _get_balanced_annotations(
        self, subsample_chunks: List[Chunk], subsample_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve or generate a balanced set of binary annotations (labels) for the given chunks.

        This internal method first attempts to load previously saved balanced annotations that match
        the current subsample. If none are found, it uses an LLM-based `EducationalEvaluator` to label
        the chunks as high-quality (1) or low-quality (0) educational content, aiming for a balanced
        number of examples per class (as defined by `self.limit_labels_per_class`). The resulting labels
        and corresponding embeddings are then saved for future reuse and returned as a balanced subset.

        Args:
            subsample_chunks: A list of Chunk objects to be annotated.
            subsample_embeddings: A NumPy array of embeddings corresponding to the chunks.

        Returns:
            A tuple of two NumPy arrays:
                - filtered_embeddings: Embeddings for chunks with valid, balanced labels.
                - filtered_labels: Binary labels (0 or 1) for those chunks.

            If no valid annotations can be collected, returns two empty arrays.
        """
        print(
            f"Getting balanced annotations ({self.limit_labels_per_class} per class)..."
        )

        # Try to load existing balanced annotations first
        result = self._load_existing_annotations(subsample_chunks, subsample_embeddings)
        if result is not None:
            return result

        # If no existing balanced annotations, create new ones
        print("No existing balanced annotations found. Creating new annotations...")

        agent_evaluator = EducationalEvaluator(
            api_url=self.llm_url,
            model_name=self.llm_model,
            limit_per_class=self.limit_labels_per_class,
        )

        all_labels = agent_evaluator.run(subsample_chunks)

        self._save_annotations(all_labels)

        result = self._extract_balanced_subset(
            all_labels, subsample_embeddings, strict=False
        )

        if result is not None:
            return result

        print("No valid annotations collected")
        return np.array([]), np.array([])

    def _load_existing_model(self, trainer: RandomForestTrainer) -> Optional[object]:
        """
        Attempt to load a pre-trained model from the specified model path using the provided trainer.

        This internal method checks whether a model file exists at `self.models_path`. If it does,
        it uses the given `RandomForestTrainer` instance to load and return the model. Otherwise,
        it returns `None`, indicating that no existing model is available.

        Args:
            trainer: An instance of `RandomForestTrainer` capable of loading a model from disk.

        Returns:
            The loaded model object if a model file exists at `self.models_path`; otherwise, `None`.
        """
        if os.path.exists(self.models_path):
            print("Loading existing trained model...")
            return trainer.load_model(self.models_path)
        return None

    def _train_new_model(
        self, subsample_embeddings: np.ndarray, labels: np.ndarray
    ) -> object:
        """
        Train a new classification model using the provided embeddings and binary labels.

        This internal method initializes a RandomForestTrainer with the instance's configuration
        parameters and trains a model on the given labeled subsample. The trained model is saved
        to disk according to the trainer's settings, and metadata may be persisted if enabled.

        Args:
            subsample_embeddings: A NumPy array of shape (n_samples, embedding_dim) containing
                                the feature vectors for the training samples.
            labels: A 1D NumPy array of binary labels (0 = low-quality, 1 = high-quality)
                    corresponding to each embedding.

        Returns:
            The trained model object (e.g., a scikit-learn RandomForestClassifier).
        """
        print("Training new model on subsample data...")
        trainer = RandomForestTrainer(
            test_size=self.test_size,
            random_state=self.random_state,
            model_path=self.models_path,
            datasets_path=self.datasets_path,
            metadata_path=self.metadata_path,
            save_metadata=self.save_metadata,
        )
        train_result = trainer.train(subsample_embeddings, labels)
        return train_result[0]

    def _get_or_train_model(
        self, all_embeddings: np.ndarray, chunks: List[Chunk]
    ) -> object:
        """
        Retrieve a pre-trained model if available; otherwise, train a new one using a balanced subsample of annotated chunks.

        This internal method first attempts to load an existing model from disk. If no model is found,
        it creates a representative subsample of the input chunks and embeddings, extracts a balanced set
        of labeled examples (high-quality vs. low-quality), validates that enough labeled data is available,
        and trains a new classifier using those examples.

        Args:
            all_embeddings: A NumPy array of shape (n_chunks, embedding_dim) containing embeddings for all input chunks.
            chunks: A list of Chunk objects corresponding to the embeddings, potentially containing annotation metadata.

        Returns:
            A trained model object (e.g., scikit-learn estimator) ready for prediction.
        """
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

        if len(filtered_labels) < self.limit_labels:
            raise ValueError(
                f"Could not collect enough balanced examples. "
                f"Got {len(filtered_labels)} total, need at least self.limit_labels"
            )

        return self._train_new_model(filtered_embeddings, filtered_labels)

    def _filter_chunks(
        self, model: object, all_embeddings: np.ndarray, chunks: List[Chunk]
    ) -> List[Chunk]:
        """
        Filter chunks based on binary predictions from a trained classification model.

        This internal method applies the provided model to a set of chunk embeddings,
        interprets the output as binary labels (0 = low-quality, 1 = high-quality),
        and returns only those chunks predicted as high-quality educational content.

        Args:
            model: A trained scikit-learn-compatible classifier with a `predict` method.
            all_embeddings: A NumPy array of shape (n_chunks, embedding_dim) containing
                            the vector representations of the input chunks.
            chunks: A list of Chunk objects corresponding one-to-one with the embeddings.

        Returns:
            A list of Chunk objects for which the model predicted label == 1.
        """
        print("Applying model to filter chunks...")

        # Get predictions for all chunks
        all_labels = model.predict(all_embeddings)
        print(f"Prediction distribution: {np.bincount(all_labels.astype(int))}")

        # Filter high-quality chunks
        return [chunk for chunk, label in zip(chunks, all_labels) if label == 1]

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
