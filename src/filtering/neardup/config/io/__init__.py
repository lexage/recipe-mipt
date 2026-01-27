import sys
import os
sys.path.insert(0, os.getcwd())

from src.filtering.neardup.config.io.input_configs import InputConfig
from src.filtering.neardup.config.io.input_configs import InputConfigType
from src.filtering.neardup.config.io.input_configs import LocalInputConfig
from src.filtering.neardup.config.io.output_configs import OutputConfig
from src.filtering.neardup.config.io.output_configs import OutputConfigType

__all__ = ["InputConfig", "InputConfigType", "LocalInputConfig", "OutputConfig", "OutputConfigType"]