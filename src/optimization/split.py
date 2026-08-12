"""Train / eval split of DS1000 for prompt optimization.

The optimizer must never see the tasks the headline PASS@1 is reported on,
otherwise the reported number is tuned-on data. A seeded sample of problem_ids
is written next to the optimization run and fed back to ``run_ds1000.py
--exclude-split`` so the benchmark run evaluates the complement.
"""

import json
import os
import random

from typing import Dict, List


def make_split(problem_ids: List, n_train: int, seed: int = 7) -> Dict[str, List]:
    """Seeded split. Sorted first so the sample does not depend on file order."""
    ordered = sorted(problem_ids, key=lambda pid: (isinstance(pid, str), pid))
    if n_train >= len(ordered):
        raise ValueError(
            f"n_train={n_train} leaves no evaluation tasks (dataset has {len(ordered)})"
        )
    rng = random.Random(seed)
    train = sorted(rng.sample(ordered, n_train),
                   key=lambda pid: (isinstance(pid, str), pid))
    train_set = set(train)
    return {
        "seed": seed,
        "n_train": n_train,
        "train": train,
        "eval": [pid for pid in ordered if pid not in train_set],
    }


def save_split(split: Dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)


def load_split(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_ids(path: str) -> List:
    return load_split(path)["train"]
