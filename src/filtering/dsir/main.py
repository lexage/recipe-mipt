# pip install data-selection

import os
import sys
sys.path.insert(0, os.getcwd())
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Dict, Callable, Iterable, Union
from tqdm import tqdm

from data_selection import HashedNgramDSIR
from data_selection.base import default_parse_example_fn

# from src.agent_constructor.core import Document
from src.agent_constructor.core import Chunk
from src.agent_constructor.chunkers import DSIRChunker

import nltk
# nltk.download('punkt_tab')


def handle_filtration(chunks: List[Chunk]) -> List[Chunk]:
    # TODO: add implementation
    return chunks


def create_load_dataset_fn(chunks: List[Chunk]) -> Callable[[str], Iterable[Dict]]:
    """
    Create the function load_dataset_fn for DSIR from the list of chunks.
    The path argument is needed for compatibility.
    """
    def load_dataset_fn(path: str) -> Iterable[Dict]:
        for chunk in chunks:
            yield {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "tokens": chunk.tokens,
                "metadata": chunk.metadata
            }
    return load_dataset_fn


def dsir_pipeline(
    raw_chunks: List[Chunk], 
    target_chunks: List[Chunk],
    num_to_sample: int,
    num_proc: Optional[int] = None,
    ngrams: int = 2,
    num_buckets: int = 10000,
    tokenizer: str = 'word_tokenize',
    min_example_length: int = 100,
    separate_targets: bool = False,
    target_proportions: Optional[List[float]] = None,
    num_tokens_to_fit: Union[str, int] = 'auto',
    top_k: bool = False
) -> List[Chunk]:
    filtered_raw_chunks = handle_filtration(raw_chunks)
    
    raw_load_fn = create_load_dataset_fn(filtered_raw_chunks)
    target_load_fn = create_load_dataset_fn(target_chunks)
    
    raw_paths = ["in_memory_raw"]
    target_paths = ["in_memory_target"]
    
    dsir = HashedNgramDSIR(
        raw_datasets=raw_paths, 
        target_datasets=target_paths, 
        cache_dir="./src/filtering/dsir/dsir_cache", 
        tokenizer=tokenizer,
        raw_load_dataset_fn=raw_load_fn,
        raw_parse_example_fn=default_parse_example_fn,
        target_load_dataset_fn=target_load_fn,
        target_parse_example_fn=default_parse_example_fn,
        num_proc=num_proc,
        ngrams=ngrams,
        num_buckets=num_buckets,
        min_example_length=min_example_length,
        separate_targets=separate_targets,
        target_proportions=target_proportions
    )
    
    dsir.fit_importance_estimator(num_tokens_to_fit=num_tokens_to_fit)
    dsir.compute_importance_weights()
    
    dsir.resample(
        out_dir="./src/filtering/dsir/resampled", 
        num_to_sample=num_to_sample, 
        cache_dir="./src/filtering/dsir/resampled_cache",
        top_k=top_k
    )
    
    resampled_chunks = []
    resampled_dir = Path("./src/filtering/dsir/resampled")
    
    for file_path in resampled_dir.glob("*.jsonl"):
        if file_path.stat().st_size == 0:
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                
                chunk = Chunk(
                    id=str(data["id"]),
                    doc_id=str(data["doc_id"]),
                    text=str(data["text"]),
                    tokens=data.get("tokens"),
                    metadata=data.get("metadata", {})
                )
                resampled_chunks.append(chunk)

    return resampled_chunks