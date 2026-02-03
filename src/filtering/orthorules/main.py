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
# from src.agent_constructor.core import Document
from src.agent_constructor.core import Chunk


def orthorules_pipeline(
    chunks: list[Chunk], 
    subset_size: int = 10000, 
    batch_size: int = 1000,
    r: int = 10,
    tau: float = 1.0,
    k: int = 20000,
    sampling_method: str = "gumbel"
) -> list[Chunk]:
    random_subset = select_random_batch(chunks, subset_size)
    data_rating(random_subset, "batch_rating_results", range(len(rules)), batch_size=batch_size)
    
    selected_rules_indices = select_rules(r=r)
    data_rating(chunks, "all_data_rating_results", selected_rules_indices, batch_size=batch_size)
    
    high_quality_data = select_high_quality_data(
        chunks, selected_rules_indices, tau=tau, k=k, sampling_method=sampling_method
    )
    
    return high_quality_data
