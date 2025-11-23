from typing import List, Dict, Any
from pydantic import BaseModel

from pydantic import BaseModel
from src.pipelines.constants import ComponentNames

# Базовый класс для всех конфигов
class ComponentConfig(BaseModel):
    type: ComponentNames
    params: Dict[str, Any] = {}
    dependencies: List[str] = []

class PipelineConfig(BaseModel):
    pipeline_type: str
    components: Dict[str, ComponentConfig]
    params: Dict[str, Any] = {}

