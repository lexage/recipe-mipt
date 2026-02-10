# pip install python-Levenshtein

import Levenshtein

def edit_similarity(
    tokens1: list[str] | list[bytes], 
    tokens2: list[str] | list[bytes]
) -> float:
    """
    Compute Edit Similarity between two token sequences.
    """
    max_len = max(len(tokens1), len(tokens2))
    if max_len == 0:
        return 1.0

    edit_dist = Levenshtein.distance(tokens1, tokens2)
    return 1.0 - (edit_dist / max_len)