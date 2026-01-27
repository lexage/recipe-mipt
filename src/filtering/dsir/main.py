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
from data_selection.base import default_load_dataset_fn, default_parse_example_fn

from src.agent_constructor.core import Document
from src.agent_constructor.chunkers import DSIRChunker

import nltk
# nltk.download('punkt_tab')


def handle_filtration(documents: List[Document]) -> List[Document]:
    # TODO: add implementation
    return documents


def split_document(args) -> List[Dict]:
    doc, chunk_length = args
    chunker = DSIRChunker(chunk_length)
    chunks = chunker.chunk(doc)
    return [{"text": chunk.text, "metadata": chunk.metadata, "id": chunk.id} for chunk in chunks]


def split_parallel(
    documents: List[Document], 
    output_file: str, 
    chunk_length: int = 128, 
    num_workers: int = None
) -> None:
    if num_workers is None:
        num_workers = cpu_count()
    
    args = [(doc, chunk_length) for doc in documents]
    
    output_str_path = f'./src/filtering/dsir/chunk_data/{output_file}'
    output_path = Path(output_str_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with Pool(processes=num_workers) as pool:
        all_chunks = list(tqdm(
            pool.imap(split_document, args),
            total=len(documents),
            desc="Documents splitting"
        ))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunks_list in all_chunks:
            for chunk in chunks_list:
                f.write(json.dumps(chunk, ensure_ascii=False).strip() + '\n')
    return output_str_path


def chunks_processing(
    filtered_raw_documents: List[Document],
    target_documents: List[Document],
    num_to_sample: int,
    chunk_length: int = 128, 
    num_workers: int = None,
    num_proc: Optional[int] = None,
    ngrams: int = 2,
    num_buckets: int = 10000,
    min_example_length: int = 100,
    separate_targets: bool = False,
    target_proportions: Optional[List[float]] = None,
    num_tokens_to_fit: Union[str, int] = 'auto',
    top_k: bool = False
) -> None:
    raw_output_path = split_parallel(
        documents=filtered_raw_documents, 
        output_file="raw.jsonl", 
        chunk_length=chunk_length, 
        num_workers=num_workers
    )
    target_output_path = split_parallel(
        documents=target_documents, 
        output_file="target.jsonl", 
        chunk_length=chunk_length, 
        num_workers=num_workers
    )
    
    dsir = HashedNgramDSIR(
        raw_datasets=[raw_output_path], 
        target_datasets=[target_output_path], 
        cache_dir="./src/filtering/dsir/dsir_cache", 
        tokenizer="word_tokenize",
        raw_load_dataset_fn=default_load_dataset_fn,
        raw_parse_example_fn=default_parse_example_fn,
        target_load_dataset_fn=default_load_dataset_fn,
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


def create_load_dataset_fn(documents: List[Document]) -> Callable[[str], Iterable[Dict]]:
    """
    Create the function load_dataset_fn for DSIR from the list of documents.
    The path argument is needed for compatibility.
    """
    def load_dataset_fn(path: str) -> Iterable[Dict]:
        for doc in documents:
            yield {
                "id": doc.id,
                "text": doc.text,
                "source": doc.source,
                "metadata": doc.metadata
            }
    return load_dataset_fn


def documents_processing(
    filtered_raw_documents: List[Document], 
    target_documents: List[Document],
    num_to_sample: int,
    num_proc: Optional[int] = None,
    ngrams: int = 2,
    num_buckets: int = 10000,
    tokenizer: str = 'wordpunct',
    min_example_length: int = 100,
    separate_targets: bool = False,
    target_proportions: Optional[List[float]] = None,
    num_tokens_to_fit: Union[str, int] = 'auto',
    top_k: bool = False
) -> None:
    raw_load_fn = create_load_dataset_fn(filtered_raw_documents)
    target_load_fn = create_load_dataset_fn(target_documents)
    
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


def dsir_pipeline(
    raw_documents: List[Document], 
    target_documents: List[Document],
    num_to_sample: int,
    is_splitting: bool = True,
    chunk_length: int = 128, 
    num_workers: int = None,
    num_proc: Optional[int] = None,
    ngrams: int = 2,
    num_buckets: int = 10000,
    tokenizer: str = 'wordpunct',
    min_example_length: int = 100,
    separate_targets: bool = False,
    target_proportions: Optional[List[float]] = None,
    num_tokens_to_fit: Union[str, int] = 'auto',
    top_k: bool = False
) -> None:
    filtered_raw_documents = handle_filtration(raw_documents)
    
    if is_splitting:
        chunks_processing(
            filtered_raw_documents=filtered_raw_documents,
            target_documents=target_documents,
            num_to_sample=num_to_sample,
            chunk_length=chunk_length, 
            num_workers=num_workers,
            num_proc=num_proc,
            ngrams=ngrams,
            num_buckets=num_buckets,
            min_example_length=min_example_length,
            separate_targets=separate_targets,
            target_proportions=target_proportions,
            num_tokens_to_fit=num_tokens_to_fit,
            top_k=top_k
        )
    else:
        documents_processing(
            filtered_raw_documents=filtered_raw_documents, 
            target_documents=target_documents,
            num_to_sample=num_to_sample,
            num_proc=num_proc,
            ngrams=ngrams,
            num_buckets=num_buckets,
            tokenizer=tokenizer,
            min_example_length=min_example_length,
            separate_targets=separate_targets,
            target_proportions=target_proportions,
            num_tokens_to_fit=num_tokens_to_fit,
            top_k=top_k
        )