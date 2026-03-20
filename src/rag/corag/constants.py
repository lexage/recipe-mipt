from enum import Enum, auto

class CoRAGSearchTypes(Enum):
    SAMPLE_SEARCH = "sample"
    TREE_SEARCH = "tree"
    BEST_OF_N_SEARCH = "best_of_n"
