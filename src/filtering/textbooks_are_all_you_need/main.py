import os
from typing import List
import numpy as np

from src.agent_constructor.core import Document
from src.utils import load_or_create_embeddings, load_or_create_annotations, filter_high_quality_documents
from src.random_forest_cls import RandomForestTrainer
from src.embeddings import Embedder

def education_value_classifier_pipeline(
    documents: List[Document],
    subsample_size: int = 1000,
    embeddings_path: str = "data/embeddings.npy",
    annotations_path: str = "data/annotations.joblib", 
    models_path: str = "data/models"
) -> List[Document]:
    """
    Pipeline for classifying and filtering documents based on their educational value.
    
    This pipeline processes a collection of documents through three main stages:
    1. Data preparation: Generates or loads embeddings for all documents
    2. Model training: Either loads a pre-trained model or trains a new classifier on a subsample
    3. Filtering: Applies the trained model to filter documents, returning only high-quality educational content
    
    The pipeline automatically manages directory creation for embeddings, annotations, and model storage.
    If a trained model already exists, it skips the training phase and uses the existing model for prediction.
    
    Args:
        documents: List of Document objects to be processed and filtered
        subsample_size: Number of documents to use for annotation and model training when no pre-trained model exists.
                       Default is 1000 documents. If the total number of documents is less than this value,
                       all documents will be used for training.
        embeddings_path: Path for storing/loading document embeddings. Default is "data/embeddings.npy".
        annotations_path: Path for storing/loading document annotations. Default is "data/annotations.joblib".
        models_path: Path for storing/loading trained classification models. Default is "models".
    
    Returns:
        List of Document objects that have been classified as high-quality educational content (label "1").
        The returned list contains a subset of the input documents that passed the quality filter.
    """
    
    # Stage 1: Data preparation
    print("=== Stage 1: Data Preparation ===")
    
    # Load or generate embeddings for all documents
    embedder = Embedder()
    all_embeddings = load_or_create_embeddings(documents, embeddings_path, embedder)
    
    # Stage 2: Model training
    print("=== Stage 2: Model Training ===")
        
    trainer = RandomForestTrainer(model_path=models_path)

    if os.path.exists(models_path):
        print("Trained model already exists. Skipping training stage.")
        model = trainer.load_model(models_path)
    else:
        print("No trained model found. Starting training stage.")
        
        # Extract a subsample for annotation and training
        if len(documents) > subsample_size:
            subsample_indices = np.random.choice(
                len(documents), size=subsample_size, replace=False
            )
            subsample_documents = [documents[i] for i in subsample_indices]
            subsample_embeddings = all_embeddings[subsample_indices]
        else:
            subsample_documents = documents
            subsample_embeddings = all_embeddings
        
        # Load or generate automatic annotations
        labels = load_or_create_annotations(subsample_documents, annotations_path)
        
        # Train the model
        model = trainer.train(subsample_embeddings, labels)
        
    
    # Stage 3: Filtering using the trained model
    print("=== Stage 3: Data Filtering ===")
    
    # Obtain classification labels for all documents
    all_labels = model.predict(all_embeddings)
    
    # Filter: return only documents labeled as "1" (high-quality)
    high_quality_documents = filter_high_quality_documents(documents, all_labels)
    
    print(f"Filtered {len(high_quality_documents)} high-quality documents "
          f"out of {len(documents)} original documents")
    
    return high_quality_documents