import importlib
from pathlib import Path
from typing import List

from src.pipelines.configs import PipelineConfig


_CONFIGS_DIR = Path(__file__).parent

def load(name: str) -> PipelineConfig:
    """Загружает pipeline_config из модуля по имени"""
    module = importlib.import_module(f"pipeline_configs.{name}")
    return module.pipeline_config

def available() -> List[str]:
    """Возвращает список доступных конфигов (имена файлов без .py)"""
    return [
        p.stem 
        for p in _CONFIGS_DIR.glob("*.py") 
        if p.stem not in ("__init__",)
    ]