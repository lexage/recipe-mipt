import sys
import os
import json
import sys
import gc
import torch
import random

# from modules.module_generation import *
sys.path.insert(0, os.getcwd())
from src.filtering.orthorules.experiments_ds1000.rating.rating_prompts import *
from src.agent_constructor.core import Document


def free_memory() -> None:
    torch.cuda.empty_cache()
    gc.collect()


def llm(questions: list[str]) -> list[int]:
    # TODO: Replace with actual LLM call
    return [0.5] * len(questions)


def rule_based_rating(rule: str, chunk_data: list[str]) -> list[int]:
    size = len(chunk_data)
    questions = []
    
    for i in range(size):
        prompt_i = rule_rating_prompt(rule, chunk_data[i])
        questions.append(prompt_i)

    free_memory()
    responses = llm(questions)
    return responses


def select_random_batch(documents: list[Document], batch_size: int = 10000) -> list[Document]:
    if len(documents) <= batch_size:
        return documents.copy()
    
    return random.sample(documents, batch_size)


def data_rating(
    data: list[Document],
    folder_name: str,
    rules_indices: list[int],
    chunk_size: int = 1000
) -> None:
    free_memory()
    
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # tokenizer_max_length=1024
    # max_new_tokens = 4
    # dtype_str='bfloat16'
    
    total_size = len(data)
    
    # Iterate over each rule
    for rule_idx in rules_indices:
        rule_folder = f'./src/filtering/orthorules/experiments_ds1000/rating/{folder_name}/rule{rule_idx}'
        
        # Create folder for each rule if it doesn't exist
        if not os.path.exists(rule_folder):
            os.makedirs(rule_folder)
                        
        for start_idx in range(0, total_size, chunk_size):
            free_memory()
            end_idx = min(start_idx + chunk_size, total_size)
            chunk_data = [d.text for d in data[start_idx:end_idx]]
            chunk_number = start_idx // chunk_size
            chunk_save_path = os.path.join(rule_folder, f'chunk{chunk_number}.json')
            
            quality_responses = rule_based_rating(rules[rule_idx], chunk_data)
            
            # Save batch results
            with open(chunk_save_path, 'w') as json_file:
                json.dump(quality_responses, json_file, indent=4)

    print("All rules processed.")
