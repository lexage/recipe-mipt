from typing import Dict, Any

from pydantic import BaseModel

from src.pipelines.constants import ComponentNames


class ComponentConfig(BaseModel):
    type: ComponentNames
    params: Dict[str, Any] = {}


class PipelineConfig(BaseModel):
    pipeline_type: str
    components: Dict[str, ComponentConfig]
    params: Dict[str, Any] = {}
