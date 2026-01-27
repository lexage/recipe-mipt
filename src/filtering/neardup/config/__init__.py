import sys
import os
sys.path.insert(0, os.getcwd())

from src.filtering.neardup.config.algorithms import AlgoConfig
from src.filtering.neardup.config.algorithms import MinHashAlgorithmConfig
from src.filtering.neardup.config.base import Config
from src.filtering.neardup.config.io import LocalInputConfig
from src.filtering.neardup.config.io import OutputConfig

__all__ = [
    "AlgoConfig",
    "Config",
    "LocalInputConfig",
    "MinHashAlgorithmConfig",
    "OutputConfig",
]