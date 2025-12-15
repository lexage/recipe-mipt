import sys
import os
import json
import sys
import gc
import torch
import random

# from modules.module_generation import *
sys.path.insert(0, os.getcwd())
from src.filtering.orthorules.experiments_ds1000.rating.rating_prompts import rules
from src.filtering.orthorules.experiments_ds1000.rating.rating_with_50rules import select_random_batch, data_rating
from src.filtering.orthorules.data_selection.dpp_select_rules import select_rules
from src.filtering.orthorules.data_selection.select_data import select_high_quality_data
from src.agent_constructor.core import Document


def orthorules_pipeline(
    documents: list[Document], 
    batch_size: int = 10000, 
    chunk_size: int = 1000,
    r: int = 10,
    tau: float = 1.0,
    k: int = 20000,
    sampling_method: str = "gumbel"
) -> list[Document]:
    random_batch = select_random_batch(documents, batch_size)
    data_rating(random_batch, "batch_rating_results", range(len(rules)), chunk_size=chunk_size)
    
    selected_rules_indices = select_rules(r=r)
    data_rating(documents, "all_data_rating_results", selected_rules_indices, chunk_size=chunk_size)
    
    high_quality_data = select_high_quality_data(
        documents, selected_rules_indices, tau=tau, k=k, sampling_method=sampling_method
    )
    
    return high_quality_data
