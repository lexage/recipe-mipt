import sys
import os
sys.path.insert(0, os.getcwd())

from typing import TypeAlias

from src.filtering.neardup.config.algorithms.base import AlgorithmConfig
from src.filtering.neardup.config.algorithms.minhash import MinHashAlgorithmConfig

AlgoConfig: TypeAlias = MinHashAlgorithmConfig

__all__ = [
    "AlgoConfig",
    "AlgorithmConfig",
    "MinHashAlgorithmConfig"
]