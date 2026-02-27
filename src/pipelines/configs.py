from typing import Dict, Any, List, Union

from pydantic import BaseModel

from src.pipelines.constants import ComponentNames, PipelinesNames


class ComponentConfig(BaseModel):
    type: ComponentNames
    params: Dict[str, Any] = {}
    deps_mapping: Dict[str, str] = {}


class PipelineConfig(BaseModel):
    type: PipelinesNames
    components: Dict[str, Union[ComponentConfig, List[ComponentConfig]]]
    params: Dict[str, Any] = {}
