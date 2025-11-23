from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field

from typing import Union
from pydantic import BaseModel
from src.pipelines.constants import ComponentNames

# Базовый класс для всех конфигов
class ComponentConfig(BaseModel):
    type: ComponentNames
    params: Dict[str, Any] = {}
    dependencies: List[str] = []

# Специфические конфиги наследуются от базового
class DBConfig(ComponentConfig):
    pass

class AgentConfig(ComponentConfig):
    pass

class RetrieverConfig(ComponentConfig):
    pass

class FilterConfig(ComponentConfig):
    pass

class ChunkerConfig(ComponentConfig):
    pass

class ContextAssemblerConfig(ComponentConfig):
    pass

AnyConfig = Union[DBConfig, AgentConfig, RetrieverConfig, FilterConfig, ChunkerConfig, ContextAssemblerConfig]


class PipelineConfig(BaseModel):
    pipeline_type: str
    components: Dict[str, AnyConfig]
    params: Dict[str, Any] = {}

