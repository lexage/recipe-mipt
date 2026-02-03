import sys
import os
sys.path.insert(0, os.getcwd())
import json
from tqdm import tqdm
import numpy as np
from typing import List
# from src.agent_constructor.core import Document
from src.agent_constructor.core import Chunk
from src.filtering.orthorules.experiments_ds1000.rating.rating_prompts import *


def average_scores(total_size: int, rule_indices: List[int], rating_results_path: str) -> List[float]:
    cumulative_sum = np.zeros(total_size)

    for rule_index in tqdm(rule_indices, desc=f"Average scores for {len(rule_indices)} rules:"):
        rule_vector = []
        rule_folder = os.path.join(rating_results_path, f'rule{rule_index}')
        
        for item in sorted(os.listdir(rule_folder)):
            file_path = os.path.join(rule_folder, item)
            with open(file_path, 'r') as file:
                scores = json.load(file)
                rule_vector.extend(scores)
        cumulative_sum += np.array(rule_vector)

    # Calculate the element-wise average
    average_vector = cumulative_sum / len(rule_indices)
    return average_vector.tolist()


def gumbel_trick_sampling(quality_scores: List[float], tau: float = 1.0, k: int = 20000) -> List[int]:
    """
    Perform quality sampling using the Gumbel top-k trick.

    Parameters:
    quality_scores (list or numpy array): List of quality scores.
    tau (float): Temperature parameter.
    top_k (int): Number of top elements to sample.

    Returns:
    list: Indices of the top-k sampled elements.
    """
    quality_scores = np.array(quality_scores)
    log_probs = quality_scores / tau
    
    # Add Gumbel noise to the log-probabilities
    noisy_scores = log_probs + np.random.gumbel(loc=0, scale=1, size=log_probs.shape)

    sorted_indices = np.argsort(noisy_scores)[-k:][::-1]
    return sorted_indices.tolist()


def stochastic_sampling(quality_scores: List[float], tau: float = 1.0, k: int = 20000) -> List[int]:
    exp_scores = np.exp(quality_scores / tau)
    probs = exp_scores / np.sum(exp_scores)
    
    selected_indices = np.random.choice(
        len(quality_scores), 
        size=k, 
        replace=False, 
        p=probs
    )
    
    return selected_indices.tolist()


def top_k_selection(quality_scores: List[float], k: int = 20000) -> List[int]:
    selected_indices = np.argsort(quality_scores)[-k:][::-1]
    return selected_indices.tolist()


def select_high_quality_data(
    chunks: List[Chunk], 
    rule_indices: List[int], 
    tau: float = 1.0, 
    k: int = 20000,
    sampling_method: str = "gumbel"
) -> List[Chunk]:
    rating_results_path = './src/filtering/orthorules/experiments_ds1000/rating/all_data_rating_results'
    total_size = len(chunks)
    
    scores = average_scores(total_size, rule_indices, rating_results_path)
    
    selected_indices = np.zeros(k)
    
    if sampling_method == "gumbel":
        selected_indices = gumbel_trick_sampling(scores, tau=tau, k=k)
    elif sampling_method == "stochastic_sampling":
        selected_indices = stochastic_sampling(scores, tau=tau, k=k)
    else:
        selected_indices = top_k_selection(scores, k=k)
    
    return [chunks[idx] for idx in selected_indices]
