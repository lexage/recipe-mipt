import os
import sys
sys.path.insert(0, os.getcwd())
# from src.agent_constructor.core import Document
from src.agent_constructor.core import Chunk
from src.filtering.qurating.prompting.score_pairwise import collect_judgement_data
from src.filtering.qurating.training.run_qurater_training import run_training
from src.filtering.qurating.data_tools.qurater_annotate import annotate_data
from src.filtering.qurating.data_tools.select_subset import select_data
from typing import List

def qurating_pipeline(
    chunks_for_training: List[Chunk],
    chunks_for_filtering: List[Chunk] 
) -> List[Chunk]:
    # getting data for QuRater training
    templates = [
        ".src/filtering/qurating/prompting/templates/pairwise_code_clarity_readability.txt",
        ".src/filtering/qurating/prompting/templates/pairwise_practical_utility_applicability.txt",
        ".src/filtering/qurating/prompting/templates/pairwise_educational_value_programming.txt",
        ".src/filtering/qurating/prompting/templates/pairwise_technical_accuracy_correctness.txt",
    ]
    output_paths = [
        ".src/filtering/qurating/datasets/judgement_data/code_clarity_readability",
        ".src/filtering/qurating/datasets/judgement_data/practical_utility_applicability",
        ".src/filtering/qurating/datasets/judgement_data/educational_value_programming",
        ".src/filtering/qurating/datasets/judgement_data/technical_accuracy_correctness"
        
    ]
    
    for template_file, output_path in zip(templates, output_paths):
        collect_judgement_data(chunks_for_training, template_file, output_path)
    
    # QuRater training
    run_training(output_paths)
    
    # getting the scores for the data we want to filter out
    annotate_data(chunks_for_filtering)
    
    # selecting data based on scores
    return select_data()