import os
import sys
sys.path.insert(0, os.getcwd())
import json
import numpy as np
from tqdm import tqdm
from dppy.finite_dpps import FiniteDPP
import numpy as np
from src.filtering.orthorules.experiments_ds1000.rating.rating_prompts import rules
from typing import List


def dpp_sample(dpp: FiniteDPP, r: int = 10) -> List[int]:
    dpp.sample_exact_k_dpp(size=r)
    selected_indices = dpp.list_of_samples[-1]
    return selected_indices

def select_rules(r: int = 10) -> List[int]:
    root_dir = './src/filtering/orthorules/experiments_ds1000/rating/batch_rating_results'
    quality_vector_ls = [] 

    for rule_idx in tqdm(range(len(rules))):
        quality_vector = []
        rule_folder = os.path.join(root_dir, f'rule{rule_idx}')
        
        # Loop through each JSON file in the rule folder
        for item in sorted(os.listdir(rule_folder)):
            input_file = os.path.join(rule_folder, item)
            with open(input_file, 'r') as f:
                num_list = json.load(f)  
            quality_vector.extend(num_list) 
        quality_vector_ls.append(quality_vector)
        
    quality_matrix = np.array(quality_vector_ls)

    # Initialize a Determinantal Point Process with the kernel matrix
    kernel_matrix = np.dot(quality_matrix, quality_matrix.T)
    dpp = FiniteDPP(kernel_type='likelihood', L=kernel_matrix)
    
    dpp_indices = dpp_sample(dpp, r)
    return sorted(dpp_indices)